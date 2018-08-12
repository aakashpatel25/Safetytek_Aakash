import logging
import traceback
from flask_restplus import Api
from restplus import settings

log = logging.getLogger(__name__)

api = Api(version='1.0', title='Rating and Recommendation Prediction API',
          description= "An API to predict rating and recommendation of the product given inoformation about \
          customer review.")

@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'
    log.exception(message)

    if not settings.FLASK_DEBUG:
        return {'message': message}, 500