import requests # import request library

url = 'http://localhost:3000/predict'

# make post request and print response
r = requests.post(
    url,
    json={
        'sepal length': 5.,
        'sepal width': 5.,
        'petal length': 5.,
        'petal width': 5.,
    },
)
print(r.json())
