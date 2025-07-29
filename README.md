# LLM as a Judge

This project explores LLMs as a judge. I've used **Gemma 2 9b** (generator LLM) and **LLaMA 3.1 8b** (critique LLM). This works in a very similar manner to GANs (Generative Adversarial Networks), where the genrator gives the initial output based on system prompt and user prompt

The critique model then evaluates this output against a criteria list, such as tone, clarity, topic. It provides both **scores** and **feedback** giving specific areas for improvement.

This feedback is not just for review, it's used to create a new, refined prompt for the generator. This iterative loop continues, with each cycle producing a more polished output, until the generated text meets a specified quality **threshold score** or the system reaches a **maximum number of iterations**. This process allows for refinement of the LLM output.

# Challenges faced

- Initially the critique model's feedback and scores werent usable and predictable, so I later on added a line to the critique prompt outlining the expected ouput format (json), fixing the issue.

- I initially wanted to display the results as they were being processed in a ```st.status``` which is a temporary element, I had to work around by showing  the ```Processing...``` text and showing the results later using the ```st.expander``` element which is a permanent element

- Based on the number of max iterations, we are making a lot of API calls to Groq, and I hit the rate limit, so I had to introduce a pause for 1 second after each iterations
