# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.

## PROBLEM STATEMENT
To implement policy evaluation on the FrozenLake-v1 environment and compare two given policies by calculating their probability of reaching the goal, average return, and state-value function, in order to identify the better policy.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)  # action chosen by policy
            
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state] * (not done))
            
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        
        if delta < theta:
            break
    
    return V
```

## OUTPUT:
### pi_frozenlake:
<img width="986" height="608" alt="image" src="https://github.com/user-attachments/assets/b2d54157-7e21-49c4-9cdf-d97ca63a9abf" />

### Reaches goal & average undiscounted return:
<img width="931" height="127" alt="image" src="https://github.com/user-attachments/assets/495fa394-5220-440d-a212-5cff749b8d3d" />

### pi_2:
<img width="777" height="663" alt="image" src="https://github.com/user-attachments/assets/a6b49f3a-22ed-4e5c-8405-4dd65c303f6e" />

### Reaches goal & average undiscounted return:
<img width="852" height="116" alt="image" src="https://github.com/user-attachments/assets/13e67ac7-bf9d-47d6-8b42-f626aabf7565" />

### Policy Comparison:
<img width="405" height="172" alt="image" src="https://github.com/user-attachments/assets/229861f9-0c4d-42a0-b704-c40c9950da9e" />

### Policy Evaluation:
<img width="535" height="199" alt="image" src="https://github.com/user-attachments/assets/04085b22-b77b-48b5-be2d-93720ab1172a" />
<img width="505" height="182" alt="image" src="https://github.com/user-attachments/assets/27d77980-f230-4351-91cf-04e9d23b10e7" />
<img width="667" height="438" alt="image" src="https://github.com/user-attachments/assets/1ee0776f-9837-4ef9-8bf7-9a6228837df6" />



## RESULT:

Therefore, policies are compared successfully using policy evaluation function in Frozen-Lake MDP.
