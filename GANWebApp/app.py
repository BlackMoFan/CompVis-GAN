from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resultGAN/', methods=['GET', 'POST'])
def resultGAN():

    import numpy as np
    # Load the latent points from the .txt file
    loaded_latent_points = np.loadtxt('./static/latent_points.txt')
    latent_points = loaded_latent_points

    # import libraries
    from keras.models import load_model
    from matplotlib import pyplot
    from numpy import load
    from numpy import mean
    from numpy import hstack
    from numpy import expand_dims
    import matplotlib.pyplot as plt
    import os
    import base64

    # average list of latent space vectors
    def average_points(points, ix):
        # convert to zero offset points
        zero_ix = [i-1 for i in ix]
        # retrieve required points
        vectors = points[zero_ix]
        # average the vectors
        avg_vector = mean(vectors, axis=0)
        # combine original and avg vectors
        # all_vectors = vstack((vectors, avg_vector))
        return avg_vector

    # create a plot of generated images
    def plot_generated(examples, rows, cols):
        # plot images
        for i in range(rows * cols):
            # define subplot
            pyplot.subplot(rows, cols, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(examples[i, :, :])

        pyplot.show()

    # load model
    model = load_model('./static/model/generator_model_100.h5')

    # Identify a few images from classes of interest
    adult_with_glasses = [18, 45, 88, 94]
    adult_no_glasses = [1, 31, 42, 46, 47, 91]
    person_with_lipstick = [12, 41, 42, 45, 50, 57, 58, 86, 97]

    #Reassign classes of interest to new variables... just to make it easy not
    # to change names all the time getting interested in new features. 
    feature1_ix = adult_with_glasses
    feature2_ix = adult_no_glasses
    feature3_ix = person_with_lipstick

    # average vectors for each class
    feature1 = average_points(latent_points, feature1_ix)
    feature2 = average_points(latent_points, feature2_ix)
    feature3 = average_points(latent_points, feature3_ix)

    # get data from the web app
    feat1 = request.form.get('feature1')
    feat2 = request.form.get('feature2')
    feat3 = request.form.get('feature3')

    # Vector arithmetic....
    # result_vector = feature1 + feature2 - feature3

    # Check the state of each checkbox and update result_vector accordingly
    if feat1 == "on":
        result_vector = feature1.copy()
    if feat2 == "on":
        result_vector = feature2.copy()
    if feat3 == "on":
        result_vector = feature3.copy()

    # Handle combinations of checkbox states
    if feat1 == "on" and feat2 == "on":
        result_vector = feature1 + feature2
    if feat1 == "on" and feat3 == "on":
        result_vector = feature1 + feature3
    if feat2 == "on" and feat3 == "on":
        result_vector = feature2 + feature3

    # Handle the case where all checkboxes are "on"
    if feat1 == feat2 == feat3 == "on":
        result_vector = feature1 + feature2 + feature3

    # Vector arithmetic....
    # result_vector = feature1 + feature2 + feature3

    # generate image using the new calculated vector
    result_vector = expand_dims(result_vector, 0)
    result_image = model.predict(result_vector)

    # scale pixel values for plotting
    result_image = (result_image + 1) / 2.0
    plt.imshow(result_image[0])
    # plt.show()

    plot_pathnb = 'static/resultGAN.png'
    if os.path.exists(plot_pathnb):
        os.remove(plot_pathnb)
    plt.savefig(plot_pathnb, format='png')

    # plt.close()


    with open(plot_pathnb, 'rb') as img_file:
        encoded_imgnb = base64.b64encode(img_file.read()).decode('utf-8')
    

    return render_template('resultGAN.html', resultGAN=encoded_imgnb)


if (__name__ == '__main__'):
    app.run()