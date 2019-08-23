from cocoa.neural.utterance import Utterance
from cocoa.neural.utterance import UtteranceBuilder as BaseUtteranceBuilder

from .symbols import markers, category_markers
from core.price_tracker import PriceScaler
from cocoa.core.entity import is_entity, Entity

class UtteranceBuilder(BaseUtteranceBuilder):
    """
    Build a word-based utterance from the batch output
    of generator and the underlying dictionaries.
    """
    def build_target_tokens(self, predictions, kb=None):
        tokens = super(UtteranceBuilder, self).build_target_tokens(predictions, kb)
        tokens = [x for x in tokens if not x in category_markers]
        return tokens

    def _entity_to_str(self, entity_token, kb):
        if entity_token[0] is None:
            return None

        raw_price = PriceScaler.unscale_price(kb, entity_token)

        if isinstance(raw_price, Entity):
            price = raw_price.canonical.value
        else:
            price = raw_price.value
        human_readable_price = "${}".format(price)
        return human_readable_price

    def get_price_number(self, entity, kb):
        raw_price = PriceScaler.unscale_price(kb, entity)
        return raw_price.canonical.value
