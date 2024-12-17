# import openai

# openai.api_key = "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"

# model = "gpt-4o-mini-2024-07-18"

# prediction = "The video opens with a close-up of a large, orange, humanoid robot with multiple eyes and limbs. The robot is in a dimly lit environment filled with smoke and debris, suggesting a chaotic or post-apocalyptic setting. The robot's body is bulky, with a rounded torso and limbs that end in what appear to be mechanical hands. It has a number \"0414\" printed on its chest. As the video progresses, more robots of similar design are revealed, each with their own unique numbers and features. They seem to be moving towards the camera, creating a sense of urgency or threat. In the background, there are glimpses of green, furry creatures that resemble monsters from popular culture, adding to the fantastical element of the scene. The lighting is low, casting shadows and giving the environment an eerie atmosphere."
# events = ["A character emerges from smoke and fire.", "The character half kneels on the ground.", "The character makes a gesture to give instructions.", "A group of characters rushes forward.", "The group of characters grabs two green monsters."]

# completion = openai.ChatCompletion.create(
#     model=model,
#     messages=[
#         {
#             "role": "user",
#             "content":
#                     "Given a video description and a list of events. For each event, classify the relationship between the video description and the event into three classes: entailment, neutral, contradiction.\n"
#                     "- \"entailment\" means that the video description entails the event.\n"
#                     "- \"contradiction\" means that some detail in the video description contradicts with the event.\n"
#                     "- \"neutral\" means that the relationship is neither \"entailment\" or \"contradiction\".\n\n"
#                     f"Video Description:\n{prediction}\n\n"
#                     f"Events: {events}\n"

#                     "Output a JSON formed as:\n"
#                     "{\n"
#                     "  \"events\": [\n"
#                     "    {\"event\": \"copy an event here\", \"relationship\": \"put class name here\",  \"reason\": \"give your reason here\"},\n"
#                     "    ...\n"
#                     "  ]\n"
#                     "}\n\n"
#                     "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Output:"
#         }
#     ]
# )
# X = completion.choices[0].message.content
# breakpoint()

# # from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
# # from transformers import LLaMAConfig, LLaMAForCausalLM
# # import torch

# # tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
# # model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
# # input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

# # out = model.generate(input_ids, max_new_tokens=10)
# # print(tokenizer.batch_decode(out))

# import transformers
# import torch
# from huggingface_hub import login

# login(token="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")

# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# temperature = 0.3

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
#     temperature = temperature,
# )
# # breakpoint()

# prediction = "A boy is adjusting a round, metal object on his head. As he turns around, his face lights up with a smile, and he looks directly at a man who has approached him. The man gently removes the round object from the boy's head. He then carefully lifts the boy off the ground. Then the man put the round object back to the boy's head."
# events = "A little boy with a helmet is looking in the mirror. His both hands are on top of his helmet adjusting its angle. A man enters from the left side of the frame and takes the helmet off the boy's head. The boy turns to the left, and the two make eye contact, after which the man lifts the boy up with both hands."
# messages=[
#     {
#         "role": "user",
#         "content":
#                 "Given a video description and a list of events. For each event, classify the relationship between the video description and the event into three classes: entailment, neutral, contradiction.\n"
#                 "- \"entailment\" means that the video description entails the event.\n"
#                 "- \"contradiction\" means that some detail in the video description contradicts with the event.\n"
#                 "- \"neutral\" means that the relationship is neither \"entailment\" or \"contradiction\".\n\n"
#                 f"Video Description:\n{prediction}\n\n"
#                 f"Events: {events}\n"

#                 "Output a JSON formed as:\n"
#                 "{\n"
#                 "  \"events\": [\n"
#                 "    {\"event\": \"copy an event here\", \"relationship\": \"put class name here\",  \"reason\": \"give your reason here\"},\n"
#                 "    ...\n"
#                 "  ]\n"
#                 "}\n\n"
#                 "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Output:"
#     }
# ]

# outputs = pipeline(
#     messages,
#     max_new_tokens=1024,
# )

# print(outputs)
# breakpoint()


from datasets import load_dataset

dataset = load_dataset("omni-research/DREAM-1K")
breakpoint()