from llm_sdk import Small_LLM_Model




def main():
    llm = Small_LLM_Model()
    str = llm.encode("hello world")
    str1 = llm.encode("hello jake and fin")
    print("those :")
    print(str)
    print(str1)

if __name__ == "__main__":
    main()