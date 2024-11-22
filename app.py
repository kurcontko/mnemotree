import asyncio

from src.memory.processor import MemoryProcessor

async def main():
    processor = MemoryProcessor()
    
    # Process an exchange
    prompt = "What are the key principles of AI safety?"
    response = "AI safety encompasses several key principles including robustness, transparency, and alignment..."
    
    memory = await processor.create_memory_from_messages(prompt, response)
    print(memory.to_str())
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())