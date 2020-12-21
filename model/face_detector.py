import tensorflow as tf


class FaceDetector(object):
    def __init__(self, args, model_path):
        super().__init__()
        self.args = args
        self.model_path = model_path
        self.model = None

    def _build(self):
        if not self.model:
            self.model = tf.saved_model.load(self.model_path)

    def __call__(self, input_x):
        """
        Given a batch of images, return the face bounding box in (x1,y1,x2,y2) format
        """

        if not self.model:
            self._build()

        boxes = []
        for sample in input_x:
            boxes.append(self.sample_call(sample))

        boxes = tf.stack(boxes, axis=0)
        boxes = boxes * self.args.resolution

        return boxes

    def sample_call(self, input_x):
        boxes = self.model.inference(tf.expand_dims(input_x, axis=0))
        boxes = tf.squeeze(boxes)
        indices, scores = \
            tf.image.non_max_suppression_with_scores(boxes[..., :4], boxes[..., 4],
                                                     max_output_size=1, iou_threshold=0.3, score_threshold=0.5)
        i = indices.numpy()[0]
        box = boxes[i, :4]
        return box
