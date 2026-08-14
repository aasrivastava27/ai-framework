from crew.crew_manager import CrewManager


google_email = "aashutoshsr27@gmail.com"

manager = CrewManager()

result = manager.run(
    route="gmail",
    question="Find my recent emails.",
    google_email=google_email,
)

print("\n================ RESULT ================\n")
print(result)