from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.inference import VariableElimination

alarm_model = BayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_burglary = TabularCPD(
    variable="Burglary", variable_card=2, values=[[0.999], [0.001]],
    state_names={"Burglary":['no','yes']},
)
cpd_earthquake = TabularCPD(
    variable="Earthquake", variable_card=2, values=[[0.998], [0.002]],
    state_names={"Earthquake":["no","yes"]},
)
cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
    state_names={"Burglary":['no','yes'], "Earthquake":['no','yes'], 'Alarm':['yes','no']},
)
cpd_johncalls = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.95, 0.1], [0.05, 0.9]],
    evidence=["Alarm"],
    evidence_card=[2],
    state_names={"Alarm":['yes','no'], "JohnCalls":['yes', 'no']},
)
cpd_marycalls = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.1, 0.7], [0.9, 0.3]],
    evidence=["Alarm"],
    evidence_card=[2],
state_names={"Alarm":['yes','no'], "MaryCalls":['yes', 'no']},
)

# Associating the parameters with the model structure
alarm_model.add_cpds(
    cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls)

alarm_infer = VariableElimination(alarm_model)

# print(alarm_infer.query(variables=["JohnCalls"],evidence={"Earthquake":"yes"}))
#
#the probability of Mary Calling given that John called

q = alarm_infer.query(variables=["Alarm", "Burglary"],evidence={"MaryCalls":"yes"})
print(q)
print('\n')

# STEP 1

def main():

    # The probability of Mary calling given that John called
    print("Probability of Mary calling given that John called")
    print("P(MaryCalls='yes' | JohnCalls='yes'):")
    print(alarm_infer.query(variables=["MaryCalls"],evidence={"JohnCalls":"yes"}))
    print('\n')

    # The probability of both John and Mary calling given Alarm
    print("Probability of both John and Mary calling given Alarm")
    print("P(JohnCalls='yes', MaryCalls='yes' | Alarm='yes'):")
    print(alarm_infer.query(variables=["JohnCalls", "MaryCalls"],evidence={"Alarm":"yes"}))
    print('\n')

    # The probability of Alarm, given that Mary called
    print("Probability of Alarm, given that Mary called")
    print("P(Alarm='yes' | MaryCalls='yes'):")
    print(alarm_infer.query(variables=["Alarm"],evidence={"MaryCalls":"yes"}))
    print('\n')

if __name__ == "__main__":
    main()