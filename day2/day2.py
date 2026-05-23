def find_average(marks):
    total = 0
    for m in marks:
        total = total + m
    
    avg = total / len(marks)
    return avg

def check_grade(avg):
    if avg >= 75:
        return "Distinction"
    elif avg >= 40:
        return "Pass"
    else:
        return "Fail"

my_marks = []
subjects = int(input("Enter number of subjects: "))

i = 0
while i < subjects:
    print("Enter marks for subject", i + 1)
    mark = int(input())
    my_marks.append(mark)
    i = i + 1

student_avg = find_average(my_marks)
student_grade = check_grade(student_avg)

print("")
print("Result")
print("All marks:", my_marks)
print("Average mark:", student_avg)
print("Final grade:", student_grade)
