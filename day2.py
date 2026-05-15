def get_average(marks_list):
    total = 0
    for m in marks_list:
        total = total + m
    
    avg = total / len(marks_list)
    return avg

def get_grade(average):
    if average >= 75:
        return "Distinction"
    elif average >= 40:
        return "Pass"
    else:
        return "Fail"

marks = []

count = int(input("How many subjects? "))

i = 0
while i < count:
    score = float(input(f"Enter score for subject {i+1}: "))
    marks.append(score)
    i = i + 1

final_avg = get_average(marks)
final_grade = get_grade(final_avg)

print("\n--- Report Card ---")
print("Marks:", marks)
print("Average:", final_avg)
print("Grade:", final_grade)
