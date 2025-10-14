import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.CombineFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;

/**
 * Gom nhiều file nhỏ thành 1 split để giảm số mapper.
 */
public class CombineWholeFileInputFormat extends CombineFileInputFormat<Text, BytesWritable> {

    @Override
    protected boolean isSplitable(JobContext context, Path file) {
        return false; // không chia nhỏ file
    }

    @Override
    public RecordReader<Text, BytesWritable> createRecordReader(
            InputSplit split, TaskAttemptContext context) throws IOException {
        return new CombineWholeFileRecordReader((CombineFileSplit) split, context);
    }

    public static class CombineWholeFileRecordReader extends RecordReader<Text, BytesWritable> {
        private CombineFileSplit split;
        private TaskAttemptContext context;
        private int index;
        private Text key = new Text();
        private BytesWritable value = new BytesWritable();

        public CombineWholeFileRecordReader(CombineFileSplit split, TaskAttemptContext context) {
            this.split = split;
            this.context = context;
            this.index = 0;
        }

        @Override
        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException {
            this.split = (CombineFileSplit) split;
            this.context = context;
            this.index = 0;
        }

        @Override
        public boolean nextKeyValue() throws IOException {
            if (index < split.getNumPaths()) {
                Path file = split.getPath(index);
                FileSystem fs = file.getFileSystem(context.getConfiguration());
                FileStatus status = fs.getFileStatus(file);

                long fileLength = status.getLen();
                if (fileLength > Integer.MAX_VALUE) {
                    throw new IOException("File too large: " + file.toString());
                }

                byte[] contents = new byte[(int) fileLength];
                try (FSDataInputStream in = fs.open(file)) {
                    IOUtils.readFully(in, contents, 0, contents.length);
                    value.set(contents, 0, contents.length);
                }

                // key = chỉ lấy tên file (000001.jpg) thay vì full path
                key.set(file.getName());
                index++;
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
            return (float) index / split.getNumPaths();
        }

        @Override
        public void close() throws IOException {
            // nothing to close
        }
    }
}
