import csv

fruits = ["apple", "banana", "mango"]
fruits.append("orange")
print(fruits)

coords = (10, 20)
print(coords)

nums = {1, 2, 2, 3}
print(nums)

student = {"name": "rajesh", "age": 20}
print(student)

grades = {}

with open("students.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        g = row[2]
        if g in grades:
            grades[g] = grades[g] + 1
        else:
            grades[g] = 1

print(grades)

with open("grade_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Grade", "Count"])
    for k, v in grades.items():
        writer.writerow([k, v])
