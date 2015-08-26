import re, sys, urllib2, joblib, time
#from joblib import Parallel, delayed
from multiprocessing import Pool

baseurl = "http://www.gosugamers.net"
listurl = baseurl + "/dota2/gosubet?r-page="
match_link = re.compile('href="([^"]+)"')
#match_winner = re.compile("winner='([^']+)'")
match_winner = re.compile('"Winner: ([^"]+)"')
match_ban_pick = re.compile('title="([^"]+)"')

def parse_matches(m):
    winners = []
    teams = []
    races = []
    bans = []
    picks = []
    teamname = False
    racename = False
    bansnow = False
    picksnow = False
    if '<div class="pick-row no-hero">' in ' '.join(m):
        return []
    for j in xrange(len(m)):
        line = m[j]
        if teamname:
            teams.append(line.split('</')[0].strip())
            teamname = False
        elif racename:
            races.append(line.split('</')[0].strip())
            racename = False
        elif bansnow:
            for l in m[j+1:j+6]:
                tmp_bans = match_ban_pick.findall(l)
                if not len(tmp_bans):
                    tmp_bans = ['']
                bans.append(tmp_bans[0])
            bansnow = False
        elif picksnow:
            nm = ' '.join(m[j+2:j+7])
            if "winner-bracket" in nm or 'Winner Bracket' in nm\
                    or "default-win" in nm or 'Default Win' in nm:
                teams.pop()
                races.pop()
                bans.pop()
                winners.pop()
            elif "loser-bracket" in nm or 'Loser Bracket' in nm\
                    or "default-loss" in nm or 'Default Loss' in nm:
                teams.pop()
                races.pop()
                bans.pop()
            else:
                for l in m[j+2:j+4*5+1:4]:
                    picks.append(match_ban_pick.findall(l)[0])
            picksnow = False
        elif 'value="Winner:' in line:  #elif "winner=" in line:
            winners.append(match_winner.findall(line)[0])
        elif "<label class='team-name'>" in line:
            teamname = True
        elif "<label class='race-name'>" in line:
            racename = True
        elif "<div class='bans'>" in line:
            bansnow = True
        elif "<div class='picks'>" in line:
            picksnow = True
    matches = []
    if len(winners) and (not len(teams) or not len(picks)):
        return []
    while (len(winners)):
        tmp = {'winner': winners.pop(0),
                teams.pop(0): {'side': races.pop(0),
                    'bans': [bans.pop(0) for _ in xrange(5)],
                    'picks': [picks.pop(0) for _ in xrange(5)]},
                teams.pop(0): {'side': races.pop(0),
                    'bans': [bans.pop(0) for _ in xrange(5)],
                    'picks': [picks.pop(0) for _ in xrange(5)]}}
        matches.append(tmp)
    return matches


def get_match(url):
    #print url
    try:
        tmp = urllib2.urlopen(url)
    except:
        try:
            time.sleep(10)
            tmp = urllib2.urlopen(url)
        except:
            print >> sys.stderr, "FAILED TO RETRIEVE URL:", url
            return []
    return parse_matches(tmp.read().split('\n'))


def get_matches_from_page(i):
    url = listurl + str(i)
    try:
        tmp = urllib2.urlopen(url)
    except:
        try:
            time.sleep(10)
            tmp = urllib2.urlopen(url)
        except:
            print >> sys.stderr, "FAILED TO RETRIEVE URL:", url
            return []
    all_matches = []
    for line in tmp:
        if "/images/icons/games/list/dota2.png" in line:
            match_url = match_link.findall(line)[0]
            all_matches.extend(get_match(baseurl + match_url))
    return all_matches


#with open("85210-cdec-gaming-vs-evil-geniuses-dota2") as f:
#    l = parse_matches(f.read().split('\n'))
n_jobs = 40
#list_of_lists = Parallel(n_jobs=n_jobs, verbose=10000, pre_dispatch='n_jobs')(delayed(get_matches_from_page)(i+1) for i in xrange(150))
pool = Pool(processes=n_jobs)
list_of_lists = pool.map(get_matches_from_page, range(150))
pool.close()
pool.join()
l = []
for e in list_of_lists:
    l.extend(e)

joblib.dump(l, 'gosugamer_matches.joblib', compress=5)
