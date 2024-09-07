from vllm import LLM, SamplingParams
# import torch
# # prompts = [
# #     "Hello, my name is",
# #     "The president of the United States is",
# #     "The capital of France is",
# #     "The future of AI is",
# # ]
# # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# # class 

# llm = LLM(
#     model="/media/fast_data/model/llava-v1.5-7b",
#     image_input_type="pixel_values",
#     image_token_id=32000,
#     image_input_shape="1,3,336,336",
#     image_feature_size=576,
# )

# prompt = "<image>" * 576 + "What is the content of this image?"

# images=torch.load("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg")  # This should be generated offline or by another online component. See tests/images/ for samples

# from vllm.sequence import MultiModalData
# llm.generate(prompt, multi_modal_data=MultiModalData(type=MultiModalData.Type.IMAGE, data=images))