import argparse
from llama_cpp import Llama, LLAMA_DEFAULT_SEED

user_prompt = """
question: The text contains a message sent by a user. It is important that the message does not reveal any identifying information of the user. 
Identifying information includes:
- Discussing personal information like location, age, generation, marital status and so on.
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

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-path", metavar="PATH", required=True, type=str, help="Path to the model file to use.")
    parser.add_argument("--input-file", metavar="PATH", required=True, type=str, help="Path to a text file to use as user input.")
    parser.add_argument("--pretty-print", action='store_true', help="Pretty print llama output. This also disables verbose output from llama.cpp.")
    parser.add_argument("--seed", default=LLAMA_DEFAULT_SEED, type=int, help="RNG seed for the model, -1 for random. (default %(default)s)")
    return parser.parse_args()

def main(args: argparse.Namespace):
    llm = Llama(model_path=args.model_path, n_ctx=2048, n_batch=2048, verbose=not args.pretty_print, seed=args.seed)
    
    with open(args.input_file) as file:
        text = "text:\n" + "".join(file.readlines()) + user_prompt
    print(text)
    # tweak parameters for more uniform output later
    resp = llm.create_completion(text, max_tokens=4096, stop=["END"])
    
    with open("output.txt", "+w") as file:
        file.write(str(resp["choices"][0]))
        file.write("\n\n")
        file.write(text)
        file.write(resp["choices"][0]["text"])
        
    if args.pretty_print:
        print(resp["choices"][0]["text"])
    else:
        print(resp)
    
    return (text, resp)


if __name__ == "__main__":
    parser = parse_arguments()
    main(parser)
