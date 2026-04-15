from pathlib import Path

from dawn.config import DawnConfig
from dawn.generator import DawnAssistant


def main() -> None:
    pdf_path = Path("../data/PEDIATRIE ALBERT ROYER-1-1.pdf")
    question = "Quand faut-il referer un enfant atteint d'anemie ?"

    config = DawnConfig(pdf_path=pdf_path)
    assistant = DawnAssistant(config)

    answer = assistant.answer(question)

    print("\nReponse DAWN\n")
    print(answer["answer"])
    print("\nSources")
    for source in answer["sources"]:
        print(f"- page {source['page']} | score={source['score']:.3f}")


if __name__ == "__main__":
    main()
