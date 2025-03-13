import json

def filter_unique_questions(input_file, output_file):
    try:
        # Load JSON data from file
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Ensure data is a list
        if not isinstance(data, list):
            raise ValueError("JSON file should contain a list of questions.")

        # Use a set to track unique 'QuestionDesc' values
        seen_descriptions = set()
        unique_questions = []

        for question in data:
            desc = question.get("QuestionDesc")  # Extract 'QuestionDesc'
            if desc and desc not in seen_descriptions:
                seen_descriptions.add(desc)
                unique_questions.append(question)

        # Save unique questions to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(unique_questions, file, indent=4, ensure_ascii=False)

        print(f"✅ Successfully saved {len(unique_questions)} unique questions to {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")

# Example usage
filter_unique_questions("mcqs.json", "unique_mcqs.json")