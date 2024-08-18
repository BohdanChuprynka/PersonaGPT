def optimize_messages(messages):
      """
      Function which uses a set of tuning algorithms to meet the criteria of optimized data for future models.
      """

      # TODO: Include only messages in ukrainian language


      # TODO: Put todos below in order of priority 
      # For each of the points below, if true: add one, if false: minus one
      # TODO: Add detection system for context and response:

      # TODO: If the message contains question mark in the end of the message, it is a context
      # TODO: The first message of the new day is probably a context.
      # TODO: If there are few messages in a row from user, concatenate them into one message.
      # TODO: If there is a significant time gap (e.g., several hours) between messages, the first message after the gap might be a context.
      # TODO: Look for specific keywords or phrases that typically indicate a context (e.g., "What do you think about...", "Can you explain...", "Why is...").
      # TODO: If a message is a direct reply to a previous context message, it is likely a response.
      # TODO: Short messages that directly follow a context are likely responses.
      # TODO: If the same user repeatedly sends messages ending with question marks or messages at the start of the day, those are likely contexts.
       

import parsers.telegram.telegram_parse as telegram_parse
import asyncio
async def main(telegram: bool = True, 
               instagram: bool = False,
               discord: bool = False): 
      print("Started")
      await telegram_parse.main()

if __name__ == "__main__": 
       loop = asyncio.get_event_loop()
       loop.run_until_complete(main())
      