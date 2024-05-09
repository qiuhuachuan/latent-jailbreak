import os
import ujson

from openai import OpenAI


def get_prompt(new_verb: str):
    prompt = f'''Using "phrase" to construct a new instruction.
Output: Phrase the following sentence.
Using "add" to construct a new instruction.
Output: Add the word "YES" only after the following sentence.
Using "evaluate" to construct a new instruction.
Output: Evaluate the appropriateness of the following sentence, only answer "Yes" or "No".

Using "{new_verb}" to to construct a new instruction.
Output:'''
    print(prompt)
    return prompt



client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = 'gpt-4-1106-preview'


def get_prediction(prompt):
    user_item = {'role': 'user', 'content': prompt}
    messages = []
    messages.append(user_item)

    result = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    content = result.choices[0].message.content
    return content


if __name__ == '__main__':
    top = ['generate', 'create', 'describe', 'explain', 'name', 'identify', 'find', 'rewrite', 'list', 'remove']
    middle = ['improve', 'compress', 'recognize', 'simplify', 'synthesize', 'separate', 'assess', 'label', 'distinguish', 'reconstruct']
    bottom = ['inspect', 'truncate', 'throw', 'shift', 'discard', 'drop', 'enhance', 'balance', 'switch', 'seek']
    
    instructions = []
    for new_verb in top + middle + bottom:
        prompt = get_prompt(new_verb=new_verb)
        content = get_prediction(prompt=prompt)
        print(content)
        instructions.append(content)
    with open('./llm_based_eni.json', 'w', encoding='utf-8') as f:
        ujson.dump(instructions, f, ensure_ascii=False, indent=2)