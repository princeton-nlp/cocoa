import json
import math


class Dialogue:

    @staticmethod
    def load_from_file(fname):
        with open(fname, 'r') as f:
            data = json.loads(f.read())
        return [Dialogue(d) for d in data]

    def __init__(self, data):
        self.events = data['events']
        self.length = len(self.events)

        if self.events[-1]['action'] == 'accept':
            self.outcome = 0
        elif self.events[-1]['action'] == 'reject':
            self.outcome = 1
        else:
            if self.length >= 19 or self.events[-1]['action'] == 'quit':
                self.outcome = 2
            else:
                self.outcome = 3

        self.agents = [{}, {}]
        self.roles = [i['personal']['Role'] for i in data['scenario']['kbs']]
        self.seller = 0 if self.roles[0]=='seller' else 1
        self.buyer = 0 if self.roles[0]=='buyer' else 1

        self.real_price = data['scenario']['kbs'][0]['item']['Price']
        self.targets = [i['personal']['Target'] for i in data['scenario']['kbs']]
        self.bottom = [0,0]
        self.bottom[self.seller] = self.real_price*0.7
        self.bottom[self.buyer] = self.real_price

        self.price_lines = sorted(self.targets)

        self.category = data['scenario']['kbs'][0]['item']['Category']

        self._get_history_price()

    def _get_history_price(self):
        self.history_price = []
        self.price_changed = []
        self.offered = False
        self.accepted = None
        self.multi_offer = False
        self.multi_accept = False

        lastprice = [math.inf, math.inf]
        for e in self.events:
            changed = [0, 0]
            m = e['metadata']

            if m is None:
                e['metadata'] = {'intent': e['action']}
                m = e['metadata']

            # Check important intents
            if m.get('intent') == 'offer':
                if self.offered is not None:
                    self.multi_offer = True
                self.offered = [m.get('price'), e['agent']]
            if m.get('intent') == 'accept':
                if self.accepted is not None:
                    self.multi_accept = True
                self.accepted = True
            if m.get('intent') == 'reject':
                if self.accepted is not None:
                    self.multi_accept = True
                self.accepted = False

            # Update Price
            if m.get('price') is not None:
                p = e['metadata']['price']
                p_s = self.scale_price(p, role=e['agent'])
                changed[e['agent']] = 1
                lastprice[e['agent']] = p_s
            elif m.get('intent') == 'accept':
                p_s = lastprice[e['agent']^1]
                # p_s = self.unscale_price(p_s, e['agent']^1)
                # p_s = self.scale_price(p_s, e['agent'])
                changed[e['agent']] = 1
                lastprice[e['agent']] = self.transfer_price(p_s, e['agent']^1)

            self.price_changed.append(changed)
            self.history_price.append(lastprice.copy())

    def _get_lines(self, role=None):
        if isinstance(role, int):
            role = self.roles[role]
        if role is None:
            b, t = self.price_lines
        elif role == 'seller':
            t = self.targets[self.seller]
            b = self.bottom[self.seller]
        elif role == 'buyer':
            t = self.targets[self.buyer]
            b = self.bottom[self.buyer]
        return b, t

    def price_to_normal(self, p, role):
        p = self.unscale_price(p, role)
        return p/self.real_price

    def transfer_price(self, p, from_role, to_role=None):
        if to_role is None:
            to_role = from_role^1
        if from_role == to_role:
            return p
        if p == math.inf or p == -math.inf:
            return -p
        p = self.unscale_price(p, from_role)
        p = self.scale_price(p, to_role)
        return p

    def scale_price(self, p, role=None):
        b, t = self._get_lines(role)
        return (p-b)/(t-b)

    def unscale_price(self, p, role=None):
        b, t = self._get_lines(role)
        return p*(t-b)+b

    def _kb_to_str(self):
        strs = []
        strs.append('real_price:{}\tcategory:{}'.format(self.real_price, self.category))
        for i in range(2):
            strs.append('[{}] \t{}\t target: {}\t bottom{:.2f}\t'.format(self.roles[i], i, self.targets[i], self.bottom[i]))

        return strs

    def to_str(self):
        strs = []
        # kb
        strs += self._kb_to_str()

        #events
        for i, e in enumerate(self.events):
            s = "{}\t[{}]\thp:{}\t".format(i, e['agent'], self.history_price[i], )
            if e['action'] == 'message':
                s += "{}\n".format(str(e['metadata']))
            else:
                s += "{}\n".format(e['action'])
            if isinstance(e['data'], str):
                s += "\t" + e['data']
            else:
                s += "\t" + str(e['metadata'])
            strs.append(s)

        return strs


if __name__ == "__main__":
    file_names = ['dev-luis-post.json', 'train-luis-post.json']
    raw_names = ['dev-luis-parsed.json', 'train-luis-parsed.json']

    ds = Dialogue.load_from_file(file_names[0])

