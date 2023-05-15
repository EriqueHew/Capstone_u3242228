import tkinter
from tkinter import *
from fitness_tools.composition.bodyfat import DurninWomersley

# Create an inter main_window
main_window: Tk = Tk()
main_window.title('Body Fat Prediction Calculator')

def calculate_body_fat():
    choice = gender_var.get()
    if choice == 'Male':
        gender = 'male'
    else:
        gender = 'female'
    #bodyfat = DurninWomersley(40,'female',(7, 5, 4, 10))
    age = int(age_entry.get())
    triceps = int(triseps_entry.get())
    biceps = int(biceps_entry.get())
    subscapular = int(subscapular_entry.get())
    suprailliac = int(suprailliac_entry.get())
    calc = DurninWomersley(age, gender, (triceps, biceps, subscapular, suprailliac))
    value = calc.siri(calc.body_density())
    display_result = ""
    result_text.delete('1.0',END)
    display_result += "\n=========================="
    display_result += "\nBody Fat Percentage (Siri)"
    display_result += "\n=========================="
    display_result += (f'\n{name_entry.get()} body fat is '
                          f'{value:.2f} percentage')
    result_text.insert("1.0",display_result)

def display_prediction():
    import bodyfat_model

# Create labels and entries for data input
name_label = Label(main_window, text="Enter your name:").grid(row=0, column=0)
name_entry = Entry(main_window,bg="light blue")
name_entry.grid(row=0, column=1)

age_label = Label(main_window, text="Age (years):").grid(row=1, column=0)
age_entry = Entry(main_window, bg="light blue")
age_entry.grid(row=1, column=1)

gender_label = Label(main_window, text="Gender:").grid(row=2, column=0)
gender_var = StringVar()
gender_var.set("Male")
gender_male = Radiobutton(main_window, text="Male", variable=gender_var, value="male")
gender_male.grid(row=2, column=1)
gender_female = Radiobutton(main_window, text="Female", variable=gender_var, value="female")
gender_female.grid(row=2, column=2)

triseps_label = Label(main_window, text="Triceps (cm):").grid(row=3, column=0)
triseps_entry = Entry(main_window, bg="light blue")
triseps_entry.grid(row=3, column=1)

briceps_label = Label(main_window, text="Biceps (cm):").grid(row=4, column=0)
biceps_entry = Entry(main_window, bg="light blue")
biceps_entry.grid(row=4, column=1)

subscapular_label = Label(main_window, text="Subscapular (cm):").grid(row=5, column=0)
subscapular_entry = Entry(main_window,bg="light blue")
subscapular_entry.grid(row=5, column=1)

suprailliac_label = Label(main_window, text="Suprailliac (cm):").grid(row=6, column=0)
suprailliac_entry = Entry(main_window,bg="light blue")
suprailliac_entry.grid(row=6, column=1)

# Create a button to calculate body density
calculate_button = Button(main_window, text="Calculate Body Fat", command=calculate_body_fat)
calculate_button.grid(row=7, column=1,sticky="w",columnspan=1)
exit_button = Button(main_window, text='Quit', command=main_window.destroy)
exit_button.grid(row=8,column=1,sticky="w",columnspan=1)
prediction_label = Button(main_window,text="Generate Body Fat prediction", command=display_prediction)
prediction_label.grid(row=9, column=1,sticky="w",columnspan=1)

# Create a text widget to display the result
result_text = Text(main_window,bg="light green",bd=3,cursor="man",width=40,height=10)
result_text.grid(row=11, column=1, columnspan=1)

# Start the inter event loop
main_window.mainloop()