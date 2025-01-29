import argparse
from src import train, evaluate, inference

def main():
    
    parser=argparse.ArgumentParser(description='Titanic Survival Prediction')
    parser.add_argument(
        '--data',
        type=str,
        help='Path to the Titanic dataset (CSV file)',
        default="data/Titanic-Dataset.csv"
    )
    
    args=parser.parse_args()

    print("Starting Model Training...")

    train.train_and_save_model(args.data)

    print("\nEvaluating Models...")
    evaluate.load_and_evaluate(args.data)

    choice=str(input("Do you want to infer: (y/n)"))

    if choice.lower()=="y":
        inference.make_predictions()
        print("\nAll tasks completed. Check results in the 'results' directory")
    elif choice.lower()=="n":
        print("\nAll tasks completed. Check results in the 'results' directory")
        pass
    else:
        print("Wrong Input\nTrain and Evaluation completed. Check results in the 'results' directory\n===Exiting===")

if __name__ == "__main__":
    main()