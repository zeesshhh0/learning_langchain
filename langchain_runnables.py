from abc import ABC, abstractmethod
import random

class Runnable(ABC):

  def __init__(self):
    print("runnable Created")

  @abstractmethod
  def invoke():
    pass


class FakeLLM (Runnable):
  def __init__(self):
    print("llm Created")

  def invoke(self, input_dict):

    response_list = [
      'hello world',
      'AGI is coming',
      'mat kar lala'
    ]

    return {'response': random.choice(response_list)}


class FakePromptTemplate(Runnable):
  def __init__(self, template, input_variables):
    print("prompt template created")
    self.template = template,
    self.input_variables = input_variables

  def invoke(self, input_dict):
    # print(self.template[0])
    # print(input_dict)
    # exit()
    return self.template[0].format(**input_dict)

class FakeStrOutputParser(Runnable):
  def __init__(self):
    print("str output created")
  
  def invoke(self, input_data):
    return input_data['response']

class RunnableConnector(Runnable):
  def __init__(self, runnable_list):
    print("runnable connector created")
    self.runnable_list = runnable_list

  def invoke(self, input_data):

    for runnable in self.runnable_list:
     
      input_data = runnable.invoke(input_data)

    return input_data


llm = FakeLLM()

prompt = FakePromptTemplate(
  template='Hello my name is {name}',
  input_variables=['name']
)

parser = FakeStrOutputParser()

chain = RunnableConnector([prompt, llm, parser])

response = chain.invoke({'name': "hellow"})

print(response)