import joblib, sys, json
import numpy as np

heroes = {}
def h2i(hero_name):
    if hero_name not in heroes:
        ind = len(heroes)/2
        heroes[hero_name] = ind
        heroes[ind] = hero_name
    return heroes[hero_name]

a = joblib.load(sys.argv[1])
x = []
y = []
xx = []
yy = []
for match in a:
    print match
    for team, bans_picks in match.iteritems():
        if team == 'winner':
            continue
        x.append([h2i(h) for h in bans_picks['picks']])
        y.append(match['winner'] == team)
x = np.array(x, dtype='uint8')
y = np.array(y, dtype='uint8')
with open("heroes.json", 'w') as wf:
    json.dump(heroes, wf)
with open("gosugamers_x.npy", 'w') as wf:
    np.save(wf, x)
with open("gosugamers_y.npy", 'w') as wf:
    np.save(wf, y)
np.savetxt("gosugamers_x.csv", x, delimiter=',')
np.savetxt("gosugamers_y.csv", y, delimiter=',')

print(len(heroes))
print(x.shape)
print(y.shape)
