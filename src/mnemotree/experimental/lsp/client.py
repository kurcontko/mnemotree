from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any

from .protocol import decode_header, encode_message, make_notification, make_request

logger = logging.getLogger(__name__)


class LspClient:
    """
    A lightweight async LSP client.
    """

    def __init__(self, command: list[str], root_dir: str):
        self.command = command
        self.root_dir = os.path.abspath(root_dir)
        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending_requests: dict[int | str, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._loop_running = False

    async def start(self) -> None:
        """Starts the language server process and the reader loop."""
        logger.info(f"Starting LSP server: {self.command} in {self.root_dir}")
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.root_dir,
        )
        self._loop_running = True
        self._reader_task = asyncio.create_task(self._read_loop())

        # Log stderr in background
        self._stderr_task = asyncio.create_task(self._log_stderr())

    async def stop(self) -> None:
        """Stops the language server."""
        self._loop_running = False

        # Cancel reader first so it stops trying to read
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task

        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

        if self.process:
            try:
                # Close stdin to signal EOF
                if self.process.stdin:
                    self.process.stdin.close()
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        await self.process.stdin.wait_closed()

                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("LSP process did not terminate, killing...")
                    try:
                        self.process.kill()
                        await self.process.wait()
                    except ProcessLookupError:
                        pass
            except ProcessLookupError:
                pass

        logger.info("LSP Client stopped")

    async def initialize(self) -> dict[str, Any]:
        """Performs the LSP initialization handshake."""
        # 1. initialize request
        params = {
            "processId": os.getpid(),
            "rootUri": f"file://{self.root_dir}",
            "capabilities": {
                "textDocument": {
                    "synchronization": {"dynamicRegistration": True, "willSave": False, "didSave": False, "willSaveWaitUntil": False},
                    "completion": {"dynamicRegistration": True, "completionItem": {"snippetSupport": False}},
                    "hover": {"dynamicRegistration": True, "contentFormat": ["markdown", "plaintext"]},
                    "signatureHelp": {"dynamicRegistration": True, "signatureInformation": {"documentationFormat": ["markdown", "plaintext"]}},
                    "definition": {"dynamicRegistration": True},
                    "references": {"dynamicRegistration": True},
                    "documentHighlight": {"dynamicRegistration": True},
                    "documentSymbol": {"dynamicRegistration": True, "symbolKind": {"valueSet": list(range(1, 27))}},
                    "codeAction": {"dynamicRegistration": True},
                    "formatting": {"dynamicRegistration": True},
                    "rangeFormatting": {"dynamicRegistration": True},
                    "rename": {"dynamicRegistration": True},
                    "publishDiagnostics": {"relatedInformation": True},
                },
                "workspace": {"applyEdit": True},
            },
            "initializationOptions": {},
            "trace": "off",
        }
        response = await self.request("initialize", params)

        # 2. initialized notification
        await self.notify("initialized", {})
        return response

    async def request(self, method: str, params: Any) -> Any:
        """Sends a request and waits for the response."""
        req_id = self._next_id()
        req_json = make_request(method, params, req_id)

        future = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = future

        await self._send(req_json)
        return await future

    async def notify(self, method: str, params: Any) -> None:
        """Sends a notification."""
        notif_json = make_notification(method, params)
        await self._send(notif_json)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send(self, content: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("LSP process not running")

        logger.debug(f"Sending: {content.get('method')} id={content.get('id')}")
        data = encode_message(content)
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self.process and self.process.stdout

        try:
            while self._loop_running:
                # Read header
                header_lines: list[bytes] = []
                while True:
                    line = await self.process.stdout.readuntil(b"\r\n")
                    if line == b"\r\n":
                        break
                    header_lines.append(line)

                header_block = b"".join(header_lines)
                content_length = decode_header(header_block)

                # Read content
                content_bytes = await self.process.stdout.readexactly(content_length)
                content = json.loads(content_bytes)

                self._handle_message(content)
        except asyncio.IncompleteReadError:
            logger.info("LSP stream ended")
        except asyncio.CancelledError:
            logger.debug("LSP read loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in LSP read loop: {e}")

    def _handle_message(self, message: dict[str, Any]) -> None:
        logger.debug(f"Received: {message.get('method')} id={message.get('id')}")
        if "id" in message:
            req_id = message["id"]
            if req_id in self._pending_requests:
                # It's a response
                future = self._pending_requests.pop(req_id)
                if "error" in message:
                    future.set_exception(RuntimeError(f"LSP Error: {message['error']}"))
                else:
                    future.set_result(message.get("result"))
            else:
                # It could be a server -> client request (not supported yet)
                # Or just a mismatched ID
                logger.debug(f"Received response/request with unknown ID: {req_id} (pending: {list(self._pending_requests.keys())})")
        else:
            # Notification
            pass
            # logger.debug(f"Received notification: {message.get('method')}")

    async def _log_stderr(self) -> None:
        assert self.process and self.process.stderr
        while self._loop_running:
            line = await self.process.stderr.readline()
            if not line:
                break
            logger.debug(f"LSP STDERR: {line.decode().strip()}")

    async def text_document_did_open(self, file_path: str, text: str, language_id: str = "python") -> None:
        """Sends textDocument/didOpen."""
        uri = f"file://{file_path}"
        params = {
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": 1,
                "text": text,
            }
        }
        await self.notify("textDocument/didOpen", params)

    async def text_document_symbol(self, file_path: str) -> list[dict[str, Any]]:
        """Sends textDocument/documentSymbol."""
        uri = f"file://{file_path}"
        params = {"textDocument": {"uri": uri}}
        result: Any = await self.request("textDocument/documentSymbol", params)
        return result

    async def text_document_references(self, file_path: str, line: int, character: int) -> list[dict[str, Any]]:
        """Sends textDocument/references."""
        uri = f"file://{file_path}"
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": False},
        }
        result: Any = await self.request("textDocument/references", params)
        return result
