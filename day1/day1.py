name = input("Enter your name: ")
age = int(input("Enter your age: "))
salary = float(input("Enter your monthly salary: "))

yearly_salary = salary * 12

skills = ["Python", "SQL"]

user_data = {
    "name": name,
    "age": age,
    "salary": salary,
    "yearly_salary": yearly_salary,
    "skills": skills
}

print("\n--- Details ---")
print("Name:", user_data["name"])
print("Age:", user_data["age"])
print("Salary:", user_data["salary"])
print("Yearly:", user_data["yearly_salary"])
print("Dictionary:", user_data)
