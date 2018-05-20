package pro.prieran.physicalactivityclassifier;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.annotation.NonNull;
import android.support.v4.util.Pair;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

public class LoneActivity extends AppCompatActivity {
    private static final String MODEL_FILE = "file:///android_asset/optimized_tfdroid.pb";
    private static final String INPUT_NODE = "I";
    private static final String OUTPUT_NODE = "O";

    private static final String[] LABELS = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"};

    private static final int COUNT_OF_VECTORS_PER_FRAME = 128;
    private static final int COUNT_OF_FEATURES_PER_VECTOR = 9;
    private static final int COUNT_OF_SERIES = 1;
    private static final int COUNT_OF_CLASSES = 6;

    @NonNull
    private final ButterworthFilter lowPassFilterX = new ButterworthFilter(0.3f, 50.0f);
    private final ButterworthFilter lowPassFilterY = new ButterworthFilter(0.3f, 50.0f);
    private final ButterworthFilter lowPassFilterZ = new ButterworthFilter(0.3f, 50.0f);

    @NonNull // 1 x [Body_acc; gyro; total_acc] x 128
    private final float[] inputFloats = new float[COUNT_OF_SERIES * COUNT_OF_VECTORS_PER_FRAME * COUNT_OF_FEATURES_PER_VECTOR];

    @NonNull
    private final float[] result = new float[COUNT_OF_CLASSES];

    /*
        Есть система координат акселерометра
            В ней вектор ускорения свободного падения направлен неизвестно куда
            Повернёшь телефон -- вектор повернётся вместе с ним

        Есть система координат, связанная с Землёй
            В ней вектор ускорения свободного падения всегда направлен вдоль отрицательного направления оси y

        Хочется сделать так, будто отрицательное направление оси 0y акселероматра направлено к центру Земли
            Для этого достаточно найти вектор ускорения свободного падения в текущий момент времени
                и повернуть всю систему координат так, что он будет направлен вдоль отрицательного направления оси y

        Поворачиваем вектор ускорения относительно оси 0x, чтобы обнулить z-составляющую
        Поворачиваем вектор ускорения относительно оси 0z, чтобы обнулить x-составляющую
     */

    private int index = 0;
    private boolean gyroReceived;
    private boolean accReceived;
    private long lastEventTime = System.currentTimeMillis();
    @NonNull
    private final SensorEventListener sensorEventListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            if (event.sensor.equals(accSensor)) {
//                Log.e("TagToSearch", "Frequency = " + (int) (1000.0f / (lastEventTime - System.currentTimeMillis())) + " Hz");
                lastEventTime = System.currentTimeMillis();

                // Направление вектора ускорения свободного падения
                inputFloats[index] = lowPassFilterX.run(event.values[0]);
                inputFloats[index + 1] = lowPassFilterY.run(event.values[1]);
                inputFloats[index + 2] = lowPassFilterZ.run(event.values[2]);

//                inputFloats[index + 3] = event.values[0];
//                inputFloats[index + 4] = event.values[1];
//                inputFloats[index + 5] = event.values[2];

                // Направление вектора ускорения смартфона
                inputFloats[index + 6] = event.values[0];
                inputFloats[index + 7] = event.values[1];
                inputFloats[index + 8] = event.values[2];

                Vector gravity = new Vector();
                gravity.x = inputFloats[index];
                gravity.y = inputFloats[index + 1];
                gravity.z = inputFloats[index + 2];

                Vector acceleration = new Vector();
                acceleration.x = inputFloats[index + 6];// - inputFloats[index];
                acceleration.y = inputFloats[index + 7];// - inputFloats[index + 1];
                acceleration.z = inputFloats[index + 8];// - inputFloats[index + 2];

//                Log.d("TagToSearch", "onSensorChanged() called with: gravity = [" + gravity + "], acceleration = [" + acceleration + "]");
                rotateIt(gravity, acceleration);

                accReceived = true;
            } else if (event.sensor.equals(gyroSensor)) {
                inputFloats[index + 3] = event.values[0];
                inputFloats[index + 4] = event.values[1];
                inputFloats[index + 5] = event.values[2];
                gyroReceived = true;
            }
//            checkIfAllDataReceived();
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }

        private void rotateIt(Vector gravity, Vector rotated) { // Ломается, когда Z > 0
            double alphaX = Math.acos(gravity.y / Math.sqrt(gravity.y * gravity.y + gravity.z * gravity.z));
            alphaX *= gravity.z < 0 ? 1 : -1;
//            Log.d("TagToSearch", "rotateIt() called with: alphaX = [" + alphaX + ")");
            rotated = Matrix.getRotatedMatrixX(alphaX).multiplyByVector(rotated);
            gravity = Matrix.getRotatedMatrixX(alphaX).multiplyByVector(gravity);
//            Log.d("TagToSearch", "rotateIt() called with: gravityX = [" + gravity + ")");
//            Log.d("TagToSearch", "rotateIt() called with: rotatedX = [" + rotated + ")");

            double gammaZ = Math.acos(gravity.y / Math.sqrt(gravity.x * gravity.x + gravity.y * gravity.y));
            gammaZ *= gravity.x > 0 ? 1: -1;
//            Log.d("TagToSearch", "rotateIt() called with: gammaZ = [" + gammaZ + ")");
            rotated = Matrix.getRotatedMatrixZ(gammaZ).multiplyByVector(rotated);
            gravity = Matrix.getRotatedMatrixZ(gammaZ).multiplyByVector(gravity);

//            Log.d("TagToSearch", "rotateIt() called with: rotatedZ = [" + rotated + ")");
            Log.d("TagToSearch", "rotateIt() called with: gravityZ = [" + gravity + ")");
        }
    };

    @NonNull
    private TensorFlowInferenceInterface inferenceInterface;
    private SensorManager sensorManager;
    private Sensor accSensor;
    private Sensor gyroSensor;
    @NonNull
    private TextView labelTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_lone);
        labelTextView = findViewById(R.id.label_text_view);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        if (sensorManager == null) {
            Toast.makeText(this, "SensorManager == null", Toast.LENGTH_SHORT).show();
            finish();
        }

        accSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
    }

    @Override
    protected void onResume() {
        super.onResume();
        registerListeners();
    }

    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(sensorEventListener);
    }

    private void checkIfAllDataReceived() {
        if (accReceived && gyroReceived) {
//        if (accReceived) {
            accReceived = false;
            gyroReceived = false;
            index += COUNT_OF_FEATURES_PER_VECTOR;
            if (index + COUNT_OF_FEATURES_PER_VECTOR > inputFloats.length) {
                sensorManager.unregisterListener(sensorEventListener);
                index = 0;

                long timeBefore = System.currentTimeMillis();
                inferenceInterface.feed(INPUT_NODE, inputFloats, COUNT_OF_SERIES, COUNT_OF_VECTORS_PER_FRAME, COUNT_OF_FEATURES_PER_VECTOR);
                inferenceInterface.run(new String[]{OUTPUT_NODE}, true);
                inferenceInterface.fetch(OUTPUT_NODE, result);
                long timeAfter = System.currentTimeMillis();

                Log.e("TagToSearch", "Time = " + (timeAfter - timeBefore) / 1000.0f + " S, out: " + Arrays.toString(result));

                final List<Pair<Float, String>> labels = new ArrayList<>(result.length); // FIXME: It should be better
                for (int i = 0; i < result.length; i++) {
                    labels.add(new Pair<>(result[i], LABELS[i]));
                }

                Collections.sort(labels, new Comparator<Pair<Float, String>>() {
                    @Override
                    public int compare(Pair<Float, String> o1, Pair<Float, String> o2) {
                        return -Float.compare(o1.first, o2.first);
                    }
                });

                float delta = labels.get(0).first - labels.get(labels.size() - 1).first;
                for (int i = 0; i < labels.size(); i++) {
                    labels.set(i, new Pair<>(labels.get(i).first / delta, labels.get(i).second));
                }

                final StringBuilder labelText = new StringBuilder();
                for (Pair<Float, String> label : labels) {
                    labelText.append(label).append("\n");
                }
                labelTextView.setText(labelText.toString());

                registerListeners();
            }
        }
    }

    private void registerListeners() {
        sensorManager.registerListener(sensorEventListener, accSensor, SensorManager.SENSOR_DELAY_GAME); // SENSOR_DELAY_GAMES means about 50Hz
        sensorManager.registerListener(sensorEventListener, gyroSensor, SensorManager.SENSOR_DELAY_GAME);
    }

    private static class Vector {
        public double x;
        public double y;
        public double z;

        @Override
        public String toString() {
            return "Vector{" +
                    "x=" + String.format(Locale.getDefault(), "%+4.4f", x) +
                    ", y=" + String.format(Locale.getDefault(), "%+4.4f", y) +
                    ", z=" + String.format(Locale.getDefault(), "%+4.4f", z) +
                    '}';
        }
    }

    private static class Matrix {
        public double[][] value;

        public static Matrix getRotatedMatrixX(double alphaX) {
            Matrix matrix = new Matrix();
            matrix.value = new double[][]{
                    {1, 0, 0},
                    {0, Math.cos(alphaX), -Math.sin(alphaX)},
                    {0, Math.sin(alphaX), Math.cos(alphaX)}
            };
            return matrix;
        }

        public static Matrix getRotatedMatrixZ(double gammaZ) {
            Matrix matrix = new Matrix();
            matrix.value = new double[][]{
                    {Math.cos(gammaZ), -Math.sin(gammaZ), 0},
                    {Math.sin(gammaZ), Math.cos(gammaZ), 0},
                    {0, 0, 1}
            };
            return matrix;
        }

        public Vector multiplyByVector(Vector vector) {
            Vector newVector = new Vector();

            newVector.x = value[0][0] * vector.x + value[0][1] * vector.y + value[0][2] * vector.z;
            newVector.y = value[1][0] * vector.x + value[1][1] * vector.y + value[1][2] * vector.z;
            newVector.z = value[2][0] * vector.x + value[2][1] * vector.y + value[2][2] * vector.z;

            return newVector;
        }
    }
}
