# get user input
user_string = input("Enter a string: ")

# Reverse the string
reversed_string = user_string[::-1]
print(f"The reversed string is: {reversed_string}")

VOWELS = "aeiouAEIOU"
vowel_count = sum(1 for char in user_string if char in VOWELS)
print(f"The number of vowels in the string is: {vowel_count}")
