from restplus.api.model.endpoints.utility import get_ml_model_data
from restplus.api.model.rating_serve import serve_rating_model
from restplus.api.model.recom_serve import serve_recom_model
import logging, cv2, numpy as np, logging, pandas as pd
from restplus.api.model.parsers import text_input
from restplus.api.apiInit import api
from flask_restplus import Resource
from flask import request
from PIL import Image

logger = logging.getLogger(__name__)

ns = api.namespace('predict', description='Rating and Recommendation prediction api given a review')

@ns.route('/rating')
class RatingPrediction(Resource):

    @api.expect(text_input)
    def post(self):
        """
        	Given a data about customer review predict the rating of the review.
            <br>
            <br>
        	@expects: text_input
        	@returns: json_probability
        """
    	logger.info('Received reqeust to predict rating of a review')
        args = dict(text_input.parse_args(request))
        ml_data = get_ml_model_data(args)
        return serve_rating_model(ml_data)


@ns.route('/recommendation')
class RatingPrediction(Resource):

    @api.expect(text_input)
    def post(self):
        """
            Given a data about customer review predict if the product was recommended by customer.
            <br>
            <br>
            @expects: text_input
            @returns: json_probability
        """
        logger.info('Received reqeust to predict a digit')
        args = dict(text_input.parse_args(request))
        ml_data = get_ml_model_data(args)
        return serve_recom_model(ml_data)