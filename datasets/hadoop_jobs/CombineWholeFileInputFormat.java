import java.io.IOException;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.CombineFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.CombineFileRecordReader;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;

/**
 * Đọc nhiều file nhỏ (ảnh) thành cặp key/value:
 * key = tên file (Text)
 * value = nội dung ảnh (BytesWritable)
 */
public class CombineWholeFileInputFormat extends CombineFileInputFormat<Text, BytesWritable> {

    @Override
    protected boolean isSplitable(JobContext context, Path file) {
        return false;
    }

    @Override
    public RecordReader<Text, BytesWritable> createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException {
        return new CombineFileRecordReader<>((CombineFileSplit) split, context, CombineWholeFileRecordReader.class);
    }

    public static class CombineWholeFileRecordReader extends RecordReader<Text, BytesWritable> {
        private CombineFileSplit split;
        private TaskAttemptContext context;
        private int index;
        private Text currentKey;
        private BytesWritable currentValue;
        private boolean processed = false;

        public CombineWholeFileRecordReader(CombineFileSplit split, TaskAttemptContext context, Integer index) {
            this.split = split;
            this.context = context;
            this.index = index;
        }

        @Override
        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException {
            // Không cần làm gì thêm
        }

        @Override
        public boolean nextKeyValue() throws IOException {
            if (processed)
                return false;

            Path file = split.getPath(index);
            FileSystem fs = file.getFileSystem(context.getConfiguration());
            FileStatus status = fs.getFileStatus(file);
            byte[] contents = new byte[(int) status.getLen()];

            try (FSDataInputStream in = fs.open(file)) {
                IOUtils.readFully(in, contents, 0, contents.length);
            }

            currentKey = new Text(file.getName());
            currentValue = new BytesWritable(contents);
            processed = true;
            return true;
        }

        @Override
        public Text getCurrentKey() {
            return currentKey;
        }

        @Override
        public BytesWritable getCurrentValue() {
            return currentValue;
        }

        @Override
        public float getProgress() {
            return processed ? 1.0f : 0.0f;
        }

        @Override
        public void close() throws IOException {
            // nothing
        }
    }
}
