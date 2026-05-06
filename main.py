from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

files = [
    "../docs/GTB_standard_Nov23.pdf",
    "../docs/GTB_platinum_Nov23.pdf",
    "../docs/GTB_gold_Nov23.pdf" 
    ]


documents = sum(
     [
         PyPDFLoader(file).load() for file in files
    ], []
     )



pieces = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documents)

recovery_db = FAISS.from_documents(
    pieces,
    embeddings
).as_retriever(search_kwargs={"k": 2})

prompt_consultation_insurance = ChatPromptTemplate.from_messages(
    [
        #("system", "Ansewer using exclusively the content of the retrieved documents. If you don't know the answer, say you don't know."),
        ("system", "Responda usando exclusivamente o conteúdo dos documentos recuperados. Se você não souber a resposta, diga que não sabe."),
        ("human", "{query}\n\n Context: \n{context}\n \n Answer: ")
    ]
)


pipeline = prompt_consultation_insurance | model | StrOutputParser()


def ansewer(question:str):
    excerpt = recovery_db.invoke(question)
    context = "\n\n".join(one_excerpt.page_content for one_excerpt in excerpt)
    return pipeline.invoke(
        {
            "query": question,
            "context": context
        }
    )

#print(ansewer("What should I do if someone stole my item?"))
print(ansewer("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold?"))
