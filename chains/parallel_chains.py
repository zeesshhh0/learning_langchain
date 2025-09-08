from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

gemini2pro = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

gemini2flash = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

summary_prompt = PromptTemplate(
  template='generate short and simple notes from this text : {text}',
  input_variables=['text']
)

quiz_prompt = PromptTemplate(
  template='generate 10 QNA from this text : {text}',
  input_variables=['text']
)

merge_prompt = PromptTemplate(
  template='merge the provided notes and QNA into single document, \n quiz -> {quiz}, notes -> {notes}',
  input_variables=['quiz', 'notes']
)

parser = StrOutputParser()

parellel_chain = RunnableParallel({
  'notes': summary_prompt | gemini2flash | parser,
  'quiz': quiz_prompt | gemini2flash | parser
})

merge_chain = merge_prompt | gemini2pro | parser

chain = parellel_chain | merge_chain 

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()