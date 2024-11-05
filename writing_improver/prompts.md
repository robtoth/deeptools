# Overview
Here are prompts for an LLM to generate code that builds our project. This file should be read and input for every change in the code base.

Also, when using this file to generate code, follow each new act of code generation with appending a description of what happened in this file. This file should continue to grow.

# Context
We are going to be build a single page web app that has a few columns. The first column will input a bunch of text. The subsequent columns will be various improvements or modifications to this text by calling LLM API's such as OpenAI or Anthropic.

# Libraries
- Flask
- NLTK

# Layers

## Layer 1: Basic Flask App
### LLM Instructions:
Create a basic Flask app that has three columns. The first column will input a bunch of text. Each subsequent column will call OpenAI's API to generate a modification of the text. The columns will perform the following tasks:
1. The first column will say "condense this text into half the number of words".
2. The second column will say "reorganize this text to be more coherent".

Since this is the very first layer, and we have no code written, you will have to stand up a flask app, create the necessary API endpoint(s) and routes, provide some basic frontend code, and read in an OpenAI API key from the api_keys.yaml file.

### LLM Response and Summary of Changes:
Created a fully functional Flask application with the following features:
- Implemented proper logging throughout the application
- Created modular functions for API key loading and OpenAI client initialization
- Set up a clean, responsive three-column layout
- Implemented text processing endpoints using OpenAI's API
- Added error handling and loading states
- Created requirements.txt with necessary dependencies

The application now allows users to:
1. Input text in the first column
2. Process it to create a condensed version (half the words)
3. Generate a reorganized, more coherent version
4. See loading states during processing
5. Handle errors gracefully

## Layer 2: 
### LLM Instructions:
Now that Layer 1 is complete, we need to process the text differently. We need to break the text, which might be extremely long, into smaller chunks, like the length of a paragraph or two, and process each one independently.

We should ideally use a known library that handles text to do this chunking instead of reinventing the wheel. The output should amalgamate the results from each chunk into a single response, with some text breaks like --------CHUNK 17--------- breaks in between the output of the chunks.

### LLM Response and Summary of Changes:
Enhanced the application with intelligent text chunking capabilities:
- Added NLTK library for text processing
- Implemented paragraph-based text chunking
- Created new functions for chunk processing:
  - `chunk_text()`: Splits text into manageable chunks
  - `process_text_in_chunks()`: Processes each chunk and combines results
- Added clear chunk separation in output with "--------CHUNK X--------" headers
- Enhanced logging for better tracking of chunk processing
- Updated requirements.txt with NLTK dependency

The application now handles long texts more effectively by:
1. Breaking input text into paragraph-based chunks
2. Processing each chunk independently
3. Combining results with clear chunk separation
4. Maintaining proper error handling and logging throughout the chunking process

## Layer 3: 
### LLM Instructions:
The chunking length is not really good. We want to figure out a good chunking length that's approximately a few paragraphs long.
Also, we have a new column to add, which is a "critique" column. This should read in the prompt instructions from critic.txt and use that, concatenated with the text, to generate a critique of the text and show it in a new column (you may need the webapp to have a horizontal scrollbar).
A problem with the first attempt at this was the estimated chunk size assumes there are newlines breaking paragraphs. Instead, we need to consider breaking this by sentence.

### LLM Response and Summary of Changes:
Enhanced the application with improved sentence-based chunking and critique functionality:

1. Improved Text Chunking:
   - Switched to NLTK's sentence tokenizer for more reliable text splitting
   - Added `get_sentences()` function to extract sentences
   - Implemented smarter chunk size estimation based on average sentence length
   - Target chunk size of ~1500 characters (3-10 sentences per chunk)
   - Maintains sentence integrity while chunking

2. Critique Feature:
   - Created new column for harsh critiques
   - Implemented critic.txt prompt loading
   - Integrated critique generation into text processing workflow

3. UI Improvements:
   - Added horizontal scrolling container
   - Fixed column widths for better readability
   - Added sticky headers for better navigation
   - Improved spacing and layout

4. Code Improvements:
   - Enhanced error handling
   - Added detailed logging
   - Improved code modularity
   - Better type hints and documentation

The key changes in this version focus on:
- Using NLTK's sentence tokenizer instead of naive newline splitting
- Smarter chunk size estimation based on actual sentence lengths
- Maintaining sentence integrity in chunks
- Better logging of chunking metrics

## Layer 4:
### LLM Instructions:
- Use tqdm to show a progress bar for the chunking processing instead of saying "Processing chunk X of Y".
- The output of the condensed chunks should be concatenated, rechunked, repeatedly until there are 500 words left.
- We need this to be beautiful and aesthetic for a writer / scribe user.
- Javascript code can be broken into a separate file endpoints.js

### LLM Response:
(TODO)
