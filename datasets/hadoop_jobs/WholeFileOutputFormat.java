import java.io.IOException;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Ghi mỗi record thành 1 file ảnh (.jpg)
 * key = tên file (vd: female_blackhair/000001.jpg)
 * value = BytesWritable (nội dung ảnh)
 */
public class WholeFileOutputFormat extends FileOutputFormat<Text, BytesWritable> {

    @Override
    public RecordWriter<Text, BytesWritable> getRecordWriter(TaskAttemptContext job) throws IOException {
        Path outputDir = getOutputPath(job);
        FileSystem fs = outputDir.getFileSystem(job.getConfiguration());

        return new RecordWriter<Text, BytesWritable>() {
            @Override
            public void write(Text key, BytesWritable value) throws IOException {
                if (key == null || value == null)
                    return;

                Path file = new Path(outputDir, key.toString());
                if (file.getParent() != null) {
                    fs.mkdirs(file.getParent());
                }

                try (FSDataOutputStream out = fs.create(file, true)) {
                    out.write(value.getBytes(), 0, value.getLength());
                }
            }

            @Override
            public void close(TaskAttemptContext context) throws IOException {
                // nothing
            }
        };
    }
}
