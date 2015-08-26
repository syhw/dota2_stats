import dota2api
import joblib
import time
import random
from joblib import Parallel, delayed
api = dota2api.Initialise(api_key='')
my_id = 76561197997856332
private_id = 4294967295  # a private profile player's 32-bit Steam ID

# keeping only very high skill (skill=3), captain mode (game_mode=2) matches

hist = api.get_match_history(account_id=my_id)#, matches_requested=1)
print(len(hist['matches']))
smid = hist['matches'][0]['match_seq_num']
print(smid)

def get_matches(cur_smid):
    l = {}
    try:
        matches = api.get_match_history_by_seq_num(start_at_match_seq_num=cur_smid)['matches']
    except:
        print("waiting on the API")
        time.sleep(random.random() * 4 + 1)
        return get_matches(cur_smid)

    for m in matches:
        try:
            d = api.get_match_details(match_id=m['match_id'])
        except:
            print("waiting on the API")
            time.sleep(random.random() * 4 + 1)
            return get_matches(cur_smid)
        if d['game_mode'] == 2: #and d['human_players'] == 10:
            mid = m['match_id']
            for p in d['players']:
                pid = p['account_id']
                if pid != private_id:
                    try:
                        tm = api.get_match_history(account_id=pid, skill=3,
                                start_at_match_id=mid, matches_requested=1)
                    except:
                        print("waiting on the API")
                        time.sleep(random.random() * 4 + 1)
                        return get_matches(cur_smid)
                    if len(tm['matches']):
                        l[mid] = d
                    break
    return l

n_jobs = 20  # 2 seems to work
list_dicts = Parallel(n_jobs=n_jobs, verbose=1)(delayed(get_matches)(i)
        for i in [smid+j*100 for j in xrange(n_jobs)])

kept_matches = list_dicts[0].copy()
for d in list_dicts[1:]:
    kept_matches.update(d)

joblib.dump(kept_matches, 'matches.joblib', compress=5)
