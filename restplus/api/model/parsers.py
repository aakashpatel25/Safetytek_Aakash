from flask_restplus import reqparse

text_input = reqparse.RequestParser()
text_input.add_argument('age', type=int, required=True, help="Age of the reviewer")
text_input.add_argument('review_title', type=str, required=False, default='', help="Review title text")
text_input.add_argument('review_text', type=str, required=False, default='' ,help="Review text")
text_input.add_argument('positive_feedback', type=int, required=True, help="Posetive feedback count")
text_input.add_argument('division_name', type=str, required=True, help="Devision name")
text_input.add_argument('class_name', type=str, required=True, help="Class name")
text_input.add_argument('dept', type=str, required=True, help="Department class")