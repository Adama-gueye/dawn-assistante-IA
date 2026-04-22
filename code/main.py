from pathlib import Path

from dawn.config import DawnConfig
from dawn.generator import DawnAssistant


def main() -> None:
    knowledge_path = Path(__file__).resolve().parent / "data"
    question = "Quand faut-il referer un enfant atteint d'anemie ?"

    config = DawnConfig(knowledge_path=knowledge_path)
    assistant = DawnAssistant(config)

    answer = assistant.answer(question)

    print("\nReponse DAWN\n")
    print(answer["answer"])
    print("\nSources")
    for source in answer["sources"]:
        print(f"- {source['source_name']} | page {source['page']} | score={source['score']:.3f}")


if __name__ == "__main__":
    main()
