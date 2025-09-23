import java.io.IOException;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

public class WholeFileInputFormat extends FileInputFormat<Text, BytesWritable> {

    @Override
    protected boolean isSplitable(JobContext context, Path filename) {
        return false; // không chia nhỏ file
    }

    @Override
    public RecordReader<Text, BytesWritable> createRecordReader(InputSplit split,
            TaskAttemptContext context) throws IOException, InterruptedException {
        WholeFileRecordReader reader = new WholeFileRecordReader();
        reader.initialize(split, context);
        return reader;
    }

    public static class WholeFileRecordReader extends RecordReader<Text, BytesWritable> {

        private Path filePath;
        private BytesWritable value = new BytesWritable();
        private Text key = new Text();
        private boolean processed = false;

        @Override
        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException {
            filePath = ((org.apache.hadoop.mapreduce.lib.input.FileSplit) split).getPath();
        }

        @Override
        public boolean nextKeyValue() throws IOException {
            if (!processed) {
                byte[] contents;
                FileSystem fs = filePath.getFileSystem(new org.apache.hadoop.conf.Configuration());
                try (FSDataInputStream in = fs.open(filePath)) {
                    contents = new byte[in.available()];
                    IOUtils.readFully(in, contents, 0, contents.length);
                    value.set(contents, 0, contents.length);
                }
                key.set(filePath.toString());
                processed = true;
                return true;
            }
            return false;
        }

        @Override
        public Text getCurrentKey() {
            return key;
        }

        @Override
        public BytesWritable getCurrentValue() {
            return value;
        }

        @Override
        public float getProgress() {
            return processed ? 1.0f : 0.0f;
        }

        @Override
        public void close() throws IOException {}
    }
}
