package pro.prieran.physicalactivityclassifier;

public class ButterworthFilter {
    private float a1, a2;
    private float b0, b1, b2;

    private float z0;  // z(k)
    private float z1;  // z(k-1)
    private float z2;  // z(k-2)
    private float y0;  // y(k);
    private float y1;  // y(k-1);
    private float y2;  // y(k-2);


    public ButterworthFilter(float cutOffFreq, float sampleFreq, float initVal) {
        float r = cutOffFreq / sampleFreq;  // fCutoff / fSample
        if (r >= 0.5) {
            System.out.println("WARNING - Butterworth filter cannot have a cutoff frequency below the Nyquist frequency!");
            r = 0.45f;  // Force a viable filter for now
        }
        double c = 1.0d / Math.tan(Math.PI * r);
        double q = Math.sqrt(2.0d);
        b0 = (float) (1.0f / (1.0f + q * c + c * c));
        b1 = 2 * b0;
        b2 = b0;
        a1 = (float) (2.0f * (c * c - 1.0f) * b0);
        a2 = (float) (-(1.0f - q * c + c * c) * b0);
        reset(initVal);
    }

    public ButterworthFilter(float cutOffFreq, float sampleFreq) {
        this(cutOffFreq, sampleFreq, Float.NaN);
    }

    public void reset(Float initVal) {
        z0 = initVal;
        z1 = initVal;
        z2 = initVal;
        y0 = initVal;
        y1 = initVal;
        y2 = initVal;
    }


    public float run(float z) {
        z2 = z1;
        z1 = z0;
        z0 = z;
        y2 = y1;
        y1 = y0;

        if (Float.isNaN(z2)) { // Then still working on initialization
            y0 = z;
        } else {  // second order butterworth filter
            y0 = b0 * z0 + b1 * z1 + b2 * z2 + a1 * y1 + a2 * y2;
        }
        return y0;
    }

    public float read() {
        return y0;
    }
}
