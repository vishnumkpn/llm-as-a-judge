import streamlit as st
import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    st.error(
        "Groq API key not found. Please set it as an environment variable 'GROQ_API_KEY' or in a .env file."
    )
    st.stop()

# draft prompt
GENERATOR_SYSTEM_PROMPT = """
You are a helpful writing assistant named Gemma. Your task is to generate text based on user requests.
You will be given a prompt and are expected to produce a high-quality, relevant response.
In subsequent turns, you may receive feedback to refine your previous output. Use this feedback to improve your next response.
"""

# Critique model prompt, describes critique behavior,returns json object with scores
CRITIQUE_SYSTEM_PROMPT = """
You are a meticulous and objective critique assistant named LLaMA.
Your purpose is to evaluate a given text based on a set of criteria.
You must provide a score for each criterion on a scale from 0.0 to 1.0, where 1.0 is a perfect score.
You must also provide meangingful feedback suggesting specific improvements.
Finally, you will suggest a 'temperature' setting (a float between 0.0 and 1.0) for the next generation cycle,
where a higher temperature encourages more creative and diverse output, and a lower temperature promotes more focused and predictable text.

You MUST respond ONLY with a single, valid JSON object. Do not add any text before or after the JSON object.
The JSON object must have the following structure:
{
  "scores": {
    "criterion_1_name": 0.0,
    "criterion_2_name": 0.0
  },
  "feedback": "Your detailed feedback and suggestions for improvement here.",
  "suggested_temperature": 0.0
}
"""

#the criteria list
def parse_list_input(input_string, data_type=str):
    """Helper function to parse comma-separated strings from text areas."""
    if not input_string:
        return []
    try:
        return [data_type(item.strip()) for item in input_string.split(',')]
    except ValueError:
        st.error(f"Error parsing list. Please ensure all items are of type {data_type.__name__} and comma-separated.")
        return []
#iteration loop for improvement
def run_iteration_loop(system_prompt, user_message, criteria_list, threshold_score, max_iters, initial_temp):
    st.session_state.history = []
    current_generator_prompt = user_message
    current_temp = initial_temp
    final_output = None
    exit_reason = ""

    for i in range(max_iters):
        iteration_num = i + 1
        st.subheader(f"Iteration {iteration_num}")
        
        iteration_data = {}

        with st.status(f"Processing Iteration {iteration_num}...", expanded=True) as status:
            # generate output for user
            status.update(label="Step 1: Generator (Gemma) is creating the draft...")
            generator_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current_generator_prompt}
            ]
            
            try:
                generator_response = client.chat.completions.create(
                    model="gemma2-9b-it",
                    messages=generator_messages,
                    temperature=current_temp,
                    max_tokens=1024,
                )
                generated_text = generator_response.choices[0].message.content
            except Exception as e:
                st.error(f"Error calling Generator API: {e}")
                status.update(label="Error in Generation", state="error")
                break

            # feedback generated
            status.update(label="Step 2: Critique (LLaMA) is evaluating the draft...")
            critique_user_prompt = f"""
            Please evaluate the following text based on these criteria: {', '.join(criteria_list)}.

            TEXT TO EVALUATE:
            ---
            {generated_text}
            ---
            Remember to respond only with the specified JSON format.
            """
            
            critique_messages = [
                {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
                {"role": "user", "content": critique_user_prompt}
            ]

            try:
                critique_response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=critique_messages,
                    temperature=0.1,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                critique_output_text = critique_response.choices[0].message.content
            except Exception as e:
                st.error(f"Error calling Critique API: {e}")
                status.update(label="Error in Critique", state="error")
                break

            # evaluate
            status.update(label="Step 3: Parsing critique and checking threshold...")
            try:
                json_match = re.search(r'\{.*\}', critique_output_text, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON object found in the critique's response.")
                
                critique_data = json.loads(json_match.group())
                scores = critique_data.get("scores", {})
                feedback = critique_data.get("feedback", "No feedback provided.")
                suggested_temp = critique_data.get("suggested_temperature", current_temp)

                score_values = [scores.get(crit, 0.0) for crit in criteria_list]
                
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Failed to parse critique's JSON response: {e}")
                st.code(critique_output_text, language="text")
                status.update(label="Error in Parsing", state="error")
                break

            # decusion
            average_score = sum(score_values) / len(score_values) if score_values else 0
            meets_threshold = average_score >= threshold_score
            
            # iteration data to be displayed
            iteration_data = {
                "iteration": iteration_num,
                "generator_prompt": current_generator_prompt,
                "generated_text": generated_text,
                "critique_scores": scores,
                "average_score": average_score,
                "critique_feedback": feedback,
                "meets_threshold": meets_threshold,
                "temperature_used": current_temp,
                "suggested_next_temp": suggested_temp
            }
            st.session_state.history.append(iteration_data)
            
            status.update(label=f"Iteration {iteration_num} complete!", state="complete")
        
        display_iteration_details(iteration_data)

        if meets_threshold:
            final_output = generated_text
            exit_reason = f"Threshold met or exceeded ({average_score:.2f} >= {threshold_score})."
            st.success(exit_reason)
            break
        
        # next loop input
        st.write("Threshold not met. Refining prompt for next iteration...")
        current_generator_prompt = f"""
        Your previous attempt was not sufficient. Please refine it based on the following feedback.
        Original User Request: "{user_message}"
        
        Previous Output:
        ---
        {generated_text}
        ---

        Critique and Suggestions for Improvement:
        ---
        {feedback}
        ---

        Please generate a new, improved version of the text that directly addresses the critique.
        """
        current_temp = suggested_temp

    # loop end
    if not final_output:
        if st.session_state.history:
             final_output = st.session_state.history[-1]["generated_text"]
        else:
            final_output = "No output was generated due to an error in the first iteration."
        exit_reason = f"Maximum iterations ({max_iters}) reached."
        st.warning(exit_reason)

    st.session_state.final_output = {
        "text": final_output,
        "reason": exit_reason
    }

def display_iteration_details(data):
    """Renders the details of a single iteration in the Streamlit UI."""
    with st.expander(f"Details for Iteration {data['iteration']}", expanded=False):
        st.info("**Generator Prompt (Gemma)**")
        st.text(data['generator_prompt'])
        st.write(f"_(Temperature: {data['temperature_used']:.2f})_")

        st.success("**Generated Text**")
        st.markdown(data['generated_text'])

        st.warning("**Critique (LLaMA)**")
        st.markdown(f"**Feedback:** {data['critique_feedback']}")
        
        cols = st.columns(2)
        cols[0].metric("Average Score", f"{data['average_score']:.2f}")
        cols[1].metric("Suggested Next Temp", f"{data['suggested_next_temp']:.2f}")

        st.write("**Individual Scores:**")
        st.json(data['critique_scores'])


# streamlit

st.set_page_config(layout="wide", page_title="LLM as a judge System")
st.title("LLM as a judge Text Refinement System")
st.markdown("Using **Gemma (Generator)** and **LLaMA (Critique)** with the Groq API for iterative text improvement.")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'final_output' not in st.session_state:
    st.session_state.final_output = None

# input sidebar
with st.sidebar:
    st.header("System Configuration")

    st.subheader("1. Core Inputs")
    user_prompt = st.text_area(
        "User Message",
        "Write a short, empathetic response to a user feeling overwhelmed with project deadlines.",
        height=100,
        help="The initial request for the Generator model."
    )

    criteria_str = st.text_area(
        "Criteria List (comma-separated)",
        "empathy, clarity, conciseness, actionable",
        help="The features the Critique model will evaluate."
    )

    st.subheader("2. Loop Control")
    threshold = st.slider(
        "Threshold Score (Average)", 0.0, 1.0, 0.85, 0.01,
        help="The average score the output must achieve to stop the loop."
    )
    max_iterations = st.number_input(
        "Max Iterations", 1, 50, 3,
        help="The maximum number of refinement cycles."
    )
    initial_temperature = st.slider(
        "Initial Temperature", 0.0, 1.0, 0.7, 0.01,
        help="The starting creativity/randomness for the Generator. This will be adjusted by the Critique."
    )

    st.subheader("3. System Behavior")
    system_prompt_gemma = st.text_area(
        "Generator (Gemma) System Prompt",
        GENERATOR_SYSTEM_PROMPT,
        height=200,
    )
    
    start_button = st.button("Start Refinement Process", type="primary")

# ui main display
if start_button:
    # parse inputs
    criteria = parse_list_input(criteria_str)
    if not user_prompt or not criteria:
        st.error("Please provide a User Message and at least one Criterion.")
    else:
        # clear previous results
        st.session_state.history = []
        st.session_state.final_output = None
        
        # main loop run
        with st.container():
            run_iteration_loop(system_prompt_gemma, user_prompt, criteria, threshold, max_iterations, initial_temperature)

# final result
if st.session_state.final_output:
    st.divider()
    st.header("Final Result")
    st.info(st.session_state.final_output['reason'])
    st.success("Final Generated Text:")
    st.markdown(st.session_state.final_output['text'])
