from crew.crew_manager import CrewManager


def main():

    crew = CrewManager()

    response = crew.run(
        "Explain Retrieval-Augmented Generation in simple terms."
    )

    print("\nResponse:\n")
    print(response)


if __name__ == "__main__":
    main()