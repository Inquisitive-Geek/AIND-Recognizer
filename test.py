try:
    t = 10*(1/0)
except ZeroDivisionError:
    for i in range(1,5):
        try:
            t = 10*(1/0)
        except ZeroDivisionError:
            print("Error Found!")
    print("Statement after 2nd try-except worked")

print("Statement after overall try-except worked")

