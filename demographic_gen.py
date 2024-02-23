import random

def demographics_factory():
    # Generate random age between 18 and 80
    age = random.randint(18, 80)

    # Generate random race from a predefined list
    races = ['White', 'Black', 'Asian', 'Hispanic', 'Other']
    race = random.choice(races)

    # Generate random location from a predefined list
    locations = ['New York', 'Tunis', 'Paris', 'London', 'Istanbul']
    location = random.choice(locations)

    # Return the generated demographic data
    return {'age': age, 'race': race, 'location': location}
