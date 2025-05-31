import sys
from llama_cpp import Llama

user_prompt = """
question: The text contains a message sent by a user. It is important that the message does not reveal any identifying information of the user. 
Identifying information includes:
- Discussing personal information like location, age, marital status and so on. 
- Location specific information such as weather, time zone, etc.
- Mentioning one's gender, tattoos, piercings, physical capacities or disabilities.
- Mentioning one's profession, studies, hobbies or involvement in activist groups.
Does the message contain any identifying information?

steps:
- list any identifying information in the text.
- explain why you identified the sections of text as de-anonymizing
- analyze the severity of said de-anonymizing information
- give an integer from 0 to 100 that describes the possible danger of the message
- answer the question and begin your answer with RESPONSE
- print END

State each step and show your work for performing that step.

1. list any identifying information in the text
"""


def main(model_path: str, args: list[str]):
    llm = Llama(model_path=model_path, n_ctx=2048, n_batch=2048)
    
    with open(args[0]) as file:
        text = "text:\n" + "".join(file.readlines()) + user_prompt
    print(text)
    # tweak parameters for more uniform output later
    resp = llm.create_completion(text, max_tokens=4096, stop=["END"])
    print(resp)
    
    with open("output.txt", "+w") as file:
        file.write(str(resp["choices"][0]))
        file.write("\n\n")
        file.write(text)
        file.write(resp["choices"][0]["text"])
        
    return (text, resp)


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2:]))
