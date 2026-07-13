from llm.groq_client import groq_client


def main():

    print("=" * 50)
    print(" AI Framework ")
    print("=" * 50)

    question = input("\nYou: ")

    answer = groq_client.ask(question)

    print("\nAssistant:\n")

    print(answer)


if __name__ == "__main__":
    main()