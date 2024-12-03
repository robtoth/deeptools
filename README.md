# deeptools
## Video #1: (8 min)
### Facial Detection Webcam Algorithm
In this video, I walk you through how to build a simple web app using Cursor that performs facial detection and tracking on a live webcam stream. I'll show you my coding process step-by-step, from setting up the app to debugging common issues like missing libraries and incorporating facial recognition using the Face API.

By the end of the 8 minutes, we end up with a web server running locally that turns on your webcam, shows the live video stream in a web page, automatically tracks all the faces, and shows an overlay on the live stream.

https://youtu.be/7Ut-dLss_qQ?si=YHeEXkw9U9dOdtbY

## Video #2: (33 min)
### Market Research App
In this video, I use Cursor AI to build, from scratch, a python script that uses the OpenAI API and Perplexity API's to do live market research on a company. By the end of this video, I end up with a standalone script that takes in a company name and uses both Perplexity and OpenAI API’s to create an up-to-date market research report of that company.

Uses cases include:
- Getting live alerts, with detailed summaries, when new information hits about a company, for investment purposes.
- Staying up-to-date on corporate competitors.
- Study and learn how a given industry works as an educational tool.
- Automatically send Slack messages or emails summarizing information about other companies.
_(One note: I was not using Perplexity’s proper “online” model, and so the information shown in the example output is not actually live data. The model name must include the word “online” for Perplexity’s API.)_

https://youtu.be/WgTImBOj5Zg?si=XEE0pscA4xwjEcch

## Video #3: (26 min)
### Deep Style Transfer Webapp
In this half hour video, I build, from scratch, a web app that takes in a picture and uses a neural network to transform that image in the style of famous artists. We allow the user to open a web browser tab, upload an image, and see that image as if painted by various artists. We use a GPU in the cloud and a deep learning model to perform the style transfer, and the results are generated in just a few seconds.

https://youtu.be/bLFl1uDgBzY?si=4DT20iQvFYvp8XIX

## Video #4/5: (38 min)
### Live Political News Aggregator with Perplexity & ChatGPT

In this video, I create a website that pulls in live political news topics (via Perplexity), parses them through conservative and liberal viewpoints (generating questions on the fly using GPT-4), performs further research on those topics from various perspectives, and spits out an aggregated summary of the day’s politics.

Imaging this could be used to automatically create a more object, unbiased automatic political news newsletter or dashboard. Let people contribute various perspectives beyond stereotypical conservative and liberal, perhaps crowdsourcing new perspectives to feed into the LLM’s. Weaving ChatGPT’s API with Perplexity’s API unlocks many possibilities for SAAS apps.

_(Video was pre-recorded in October 2024)_

https://youtu.be/EkD0aqhlvlI?si=Yc8VCrEW6_gSdFQj

## Video #5/5: (34 min)
### Writing Helper AI App
In this 30 minute video, I use Cursor and the OpenAI API to build a webapp that inputs some long text such as paper or blog post or article, and then runs it through a few different prompts to both improve and critique the text. This time, I take the approach of writing out my code generation prompts in a file that I keep adding to as I build. This allows Cursor to have context of how the app has been built, layer by layer, and what it's meant to do.

https://youtu.be/EpI5YPQPFz4?si=xsUkw6MBPyVCCWfD
