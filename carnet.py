from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("KeyPresent","Starts"),  # added node
])

# Defining the parameters using CPT

cpd_keypresent = TabularCPD(        # added CPT
    variable="KeyPresent", variable_card=2,
    values=[[0.70], [0.30]],
    state_names={"KeyPresent": ["yes", "no"]}
)



cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
           ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2 ,2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent": ["yes", "no"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keypresent)  # added cpd_keypresent

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))


# STEP 2

def main():

    # Given that the car will not move, what is the probability that the battery is not working
    print("Given that the car will not move, what is the probability that the battery is not working")
    print("P(Battery='Doesn't work' | Moves='no'):")
    print(car_infer.query(variables=["Battery"],evidence={"Moves":"no"}))
    print('\n')

    # Given that the radio is not working, what is the probability that the car will not start
    print("Given that the radio is not working, what is the probability that the car will not start")
    print("P(Starts='no' | Radio=\"Doesn't turn on\"):")
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"Doesn't turn on"}))
    print('\n')

    # Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it
    print("Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it")
    print("P(Radio | Battery='Works') vs P(Radio | Battery='Works', Gas='Full'):")
    print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works"}))
    print('\n')
    print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works", "Gas":"Full"}))
    print('\n')
    print("yes, the probability changes")

    # Given that the car doesn't move, how does the probability of the ignition failing change if weobserve that the car does not have gas in it
    print("Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas in it")
    print("P(Ignition=\"Doesn't work\" | Moves='no') vs P(Ignition=\"Doesn't work\" | Moves='no', Gas='Empty'):")
    print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no"}))
    print('\n')
    print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no", "Gas":"Empty"}))
    print('\n')
    print("When the gas is known to be empty, the probability of ignition failing increases.")

    # What is the probability that the car starts if the radio works and it has gas in it
    print("What is the probability that the car starts if the radio works and it has gas in it")
    print("P(Starts='yes' | Radio='turns on', Gas='Full'):")
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"turns on", "Gas":"Full"}))
    print('\n')

    # STEP 3

    # probability that the key is not present given that the car does not move
    print("Probability that the key is not present given that the car does not move")
    print("P(KeyPresent='no' | Moves='no'):")
    print(car_infer.query(variables=["KeyPresent"],evidence={"Moves":"no"}))
    print('\n')

if __name__ == "__main__":
    main()