import dota2api
import joblib
import time
api = dota2api.Initialise(api_key='')
my_id = 76561197997856332
private_id = 4294967295  # a private profile player's 32-bit Steam ID

hist = api.get_match_history(account_id=my_id)#, matches_requested=1)
print(len(hist['matches']))
cur_smid = hist['matches'][0]['match_seq_num']  # 0 = newest, -1 = oldest
print(cur_smid)

# keeping only very high skill (skill=3), captain mode (game_mode=2) matches
kept_matches = {}

def get_matches(cur_smid):
    try:
        matches = api.get_match_history_by_seq_num(start_at_match_seq_num=cur_smid,
                game_mode=2)['matches']
    except:
        print("waiting on the API")
        time.sleep(3) # wait 3 seconds
        return get_matches(cur_smid)
    for m in matches:
        cur_smid = m['match_seq_num']
        try:
            d = api.get_match_details(match_id=m['match_id'])
        except:
            print("waiting on the API")
            time.sleep(3) # wait 3 seconds
            return get_matches(cur_smid)
        if d['game_mode'] == 2: #and d['human_players'] == 10:
            mid = m['match_id']
            if mid not in kept_matches:
                for p in d['players']:
                    pid = p['account_id']
                    if pid != private_id:
                        try:
                            tm = api.get_match_history(account_id=pid, skill=3,
                                    start_at_match_id=mid, matches_requested=1)
                        except:
                            print("waiting on the API")
                            time.sleep(3) # wait 3 seconds
                            return get_matches(cur_smid)
                        if len(tm['matches']):
                            kept_matches[mid] = d
                        break
    return cur_smid + 1

for _ in xrange(1000):
    cur_smid = get_matches(cur_smid)
    joblib.dump(kept_matches, 'matches.joblib', compress=5)
    if cur_smid <= 0:
        break
    print("dumped", len(kept_matches), "so far")
