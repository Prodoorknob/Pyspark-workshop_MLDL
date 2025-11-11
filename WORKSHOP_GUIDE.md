# Databricks Performance Pitfalls Workshop Guide

## 📋 Workshop Overview

**Duration:** 70 minutes  
**Target Audience:** Novice to intermediate Spark/Databricks users  
**Format:** Hands-on, interactive notebook-based workshop  
**Prerequisites:** Basic Python and SQL knowledge

---

## 🎯 Learning Objectives

By the end of this workshop, participants will be able to:

1. Identify and fix common Spark performance pitfalls
2. Read and interpret Spark execution plans and UI
3. Apply optimization techniques for joins, aggregations, and transformations
4. Implement production-ready error handling and monitoring
5. Schedule and orchestrate Databricks jobs
6. Set up alerting and retry strategies

---

## 📚 Workshop Materials

### Main Notebooks:

1. **`Databricks_Performance_Pitfalls_Workshop.ipynb`** (60 mins)
   - Core performance pitfalls with hands-on examples
   - Integrated job scheduling overview
   - Production pipeline example

2. **`Advanced_Topics_Job_Orchestration.ipynb`** (Optional - post-workshop)
   - Deep dive into Delta Lake
   - File formats and compression
   - Advanced orchestration patterns
   - Production job templates

---

## ⏱️ Detailed Schedule (70 minutes)

### Introduction (5 minutes)
- Workshop objectives
- Environment setup verification
- Dataset overview

### Core Topics (50 minutes)

#### **Pitfall #1: Shuffle Explosion** (7 mins)
- What are shuffles and why they're expensive
- Identifying shuffles in execution plans
- Minimizing unnecessary shuffles
- **Hands-on:** Compare bad vs. good shuffle patterns

#### **Pitfall #2: Skewed Keys** (7 mins)
- Understanding data skew
- Impact on performance
- Detection techniques
- Solutions: AQE, salting, broadcast joins
- **Hands-on:** Fix skewed join example

#### **Pitfall #3: Join Strategy Not Optimized** (6 mins)
- Types of joins: Broadcast vs. Sort-Merge
- When to use each strategy
- Using broadcast hints
- **Hands-on:** Optimize join with broadcast hint

#### **Pitfall #4: Python UDF Slowness** (7 mins)
- Why Python UDFs are slow
- Native Spark functions vs. UDFs
- Pandas UDFs as alternative
- **Hands-on:** Performance comparison

#### **Pitfall #5: Ineffective Caching** (6 mins)
- When to cache (and when not to)
- Storage levels explained
- Cache before or after filtering?
- **Hands-on:** Proper caching patterns

#### **Pitfall #6: Timezone/Timestamp Issues** (6 mins)
- Common timestamp parsing pitfalls
- Timezone awareness
- Native date functions
- **Hands-on:** Fix timestamp handling

#### **Pitfall #7: Bad Partitioning** (7 mins)
- Partition sizing guidelines
- Physical vs. logical partitioning
- repartition() vs. coalesce()
- **Hands-on:** Optimize partition strategy

#### **Pitfall #8: Not Reading Plans/UI** (7 mins)
- How to read execution plans
- Navigating Spark UI
- Identifying bottlenecks
- **Hands-on:** Analyze complex query plan

### Production Pipeline Example (8 minutes)
- Putting it all together
- Production-ready code
- Monitoring and metrics
- **Hands-on:** Run complete pipeline

### Job Scheduling Overview (10 minutes)
- Configuring Databricks Jobs
- Schedule patterns (cron)
- Retry and timeout settings
- Alerting configuration
- **Demo:** Job configuration walkthrough

### Wrap-up & Q&A (7 minutes)
- Key takeaways summary
- Best practices checklist
- Resources for further learning
- Questions and discussion

---

## 🚀 Getting Started

### Step 1: Import Notebooks

1. Log into your Databricks workspace
2. Go to **Workspace** → **Create** → **Import**
3. Upload both notebooks:
   - `Databricks_Performance_Pitfalls_Workshop.ipynb`
   - `Advanced_Topics_Job_Orchestration.ipynb`

### Step 2: Attach to Cluster

1. Open the main workshop notebook
2. Click **Connect** dropdown
3. Select an existing cluster or create a new one:
   - **Recommended:** Databricks Runtime 12.2 LTS or higher
   - **Node type:** Standard_DS3_v2 (or equivalent)
   - **Workers:** 2-4 nodes
   - **Enable autoscaling:** Yes

### Step 3: Verify Setup

Run the setup cells to verify:
- Spark session initialized
- Sample data created successfully
- Delta Lake enabled (for advanced notebook)

---

## 🎓 Teaching Tips

### For Instructors:

**Pacing:**
- Stick to the time allocations
- Use a timer for each section
- Leave buffer time for questions

**Engagement:**
- Encourage participants to run cells as you go
- Ask participants to share what they see in outputs
- Use Zoom polls or raise-hand features for quick checks

**Troubleshooting:**
- Have TAs available for 1-on-1 help
- Create a Slack channel for questions
- Prepare common error solutions

### Common Issues:

1. **Cluster startup time:**
   - Start cluster 10 minutes before workshop
   - Have participants attach early

2. **Cell execution delays:**
   - Pre-run cells that create large datasets
   - Use smaller datasets for demos if needed

3. **Different Databricks versions:**
   - Workshop tested on DBR 12.2+
   - Some outputs may vary slightly

---

## 📊 Topics Covered by Notebook

### Main Workshop Notebook:
- ✅ Shuffle Explosion (wide transformations)
- ✅ Skewed Keys
- ✅ Join Strategy Not Optimized
- ✅ Python UDF Slowness
- ✅ Ineffective Caching/Persistence
- ✅ Timezone / Timestamp Parsing Pitfalls
- ✅ Bad Partitioning Strategy
- ✅ Not Reading Plans and UI
- ✅ Job Scheduling Basics
- ✅ Alerting/Retry Strategy Overview

### Advanced Topics Notebook:
- ✅ Delta/Transactional Issues (Concurrency & Retention)
- ✅ Object Store Semantics (S3/GCS/Azure)
- ✅ Inefficient Formats / Codecs
- ✅ Serialization Overhead
- ✅ Advanced Job Orchestration Patterns

---

## 🎯 Success Metrics

Participants should be able to:

- [ ] Explain why shuffles are expensive and how to minimize them
- [ ] Identify data skew and apply appropriate solutions
- [ ] Choose the right join strategy for different scenarios
- [ ] Avoid Python UDFs and use native Spark functions
- [ ] Apply proper caching strategies
- [ ] Handle timestamps correctly with timezone awareness
- [ ] Determine appropriate partition sizing
- [ ] Read and interpret Spark execution plans
- [ ] Navigate and use Spark UI for debugging
- [ ] Configure a basic Databricks job with scheduling
- [ ] Implement retry logic and error handling

---

## 📖 Post-Workshop Resources

### Documentation:
- [Databricks Documentation](https://docs.databricks.com)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Delta Lake Documentation](https://docs.delta.io)

### Books:
- "Spark: The Definitive Guide" by Bill Chambers & Matei Zaharia
- "Learning Spark, 2nd Edition" by Jules Damji et al.

### Online Courses:
- Databricks Academy (free courses available)
- Coursera: Big Data Analysis with Scala and Spark
- Udemy: Apache Spark courses

### Blogs & Articles:
- [Databricks Blog](https://databricks.com/blog)
- [Netflix Tech Blog - Spark Articles](https://netflixtechblog.com)
- [Uber Engineering Blog - Big Data](https://eng.uber.com/category/articles/big-data/)

### Community:
- [Databricks Community Forums](https://community.databricks.com)
- [Stack Overflow - apache-spark tag](https://stackoverflow.com/questions/tagged/apache-spark)
- [Apache Spark User Mailing List](https://spark.apache.org/community.html)

---

## 🔧 Advanced Workshop Customization

### For Different Audiences:

**Beginners:**
- Spend more time on fundamentals
- Skip advanced patterns (salting, circuit breakers)
- Focus on UI interpretation
- Extend to 90 minutes

**Intermediate/Advanced:**
- Move faster through basics
- Deep dive into execution plans
- Add complex optimization scenarios
- Include Advanced Topics notebook

**Specific Industry Focus:**
- Add domain-specific examples
- Use relevant datasets from your industry
- Include compliance/governance topics for regulated industries

### Custom Datasets:

Replace sample data with your own:

```python
# In the setup cells, replace with:
transactions_df = spark.read.format('delta').load('dbfs:/your/data/path')
customers_df = spark.read.format('delta').load('dbfs:/your/customers/path')
```

---

## ✅ Production Readiness Checklist

Use this checklist for your actual production jobs:

### Performance:
- [ ] Reviewed execution plan with `explain()`
- [ ] Checked Spark UI for bottlenecks
- [ ] Optimized join strategies
- [ ] Validated partition sizing
- [ ] Removed unnecessary caching
- [ ] Eliminated Python UDFs

### Reliability:
- [ ] Implemented error handling
- [ ] Added retry logic
- [ ] Set appropriate timeouts
- [ ] Configured alerts
- [ ] Prevented concurrent runs
- [ ] Added comprehensive logging

### Data Quality:
- [ ] Input validation checks
- [ ] Null/duplicate detection
- [ ] Business logic validation
- [ ] Record count thresholds
- [ ] Output data verification

### Monitoring:
- [ ] Success/failure alerts configured
- [ ] Job duration tracking
- [ ] Data volume monitoring
- [ ] Quality metrics dashboard
- [ ] SLA monitoring

### Documentation:
- [ ] Job purpose documented
- [ ] Dependencies listed
- [ ] Runbook created
- [ ] Contact information added
- [ ] Troubleshooting guide

---

## 🆘 Troubleshooting Guide

### Issue: "Cluster won't start"
**Solution:**
- Check cluster configuration
- Verify permissions
- Try different runtime version
- Contact workspace admin

### Issue: "Cells run slowly"
**Solution:**
- Reduce data size for demos
- Check cluster utilization
- Review Spark UI for bottlenecks
- Add more workers if needed

### Issue: "Import notebooks fails"
**Solution:**
- Try downloading and re-uploading
- Check file format (.ipynb)
- Verify workspace permissions
- Import to different folder

### Issue: "Sample data creation errors"
**Solution:**
- Check available storage
- Verify write permissions
- Reduce data size
- Use alternative paths

### Issue: "Delta Lake not available"
**Solution:**
- Ensure DBR 8.0+ runtime
- Check cluster configuration
- Verify Delta Lake is enabled
- Restart cluster

---

## 📞 Support & Feedback

### During Workshop:
- Use dedicated Slack channel
- Raise hand for immediate help
- Type questions in chat
- Contact TAs for 1-on-1 assistance

### After Workshop:
- Email: [your-team-email]
- Slack: #databricks-help
- Office hours: [schedule]
- Feedback form: [link]

---

## 🎉 Workshop Variants

### 90-Minute Extended Version:
Add these topics:
- Z-ordering in Delta Lake
- Bloom filters
- Dynamic file pruning
- Advanced monitoring dashboards

### 3-Hour Deep Dive:
Include:
- Both notebooks in full
- Live debugging session
- Hands-on job creation
- Custom exercise with participant data

### Half-Day Masterclass:
- Full workshop content
- Advanced optimization techniques
- Real-world case studies
- Group exercise: optimize actual production job
- Q&A and consulting time

---

## 📝 Feedback Template

Please provide feedback after the workshop:

**Content:**
- Was the content relevant? (1-5)
- Was the difficulty level appropriate? (1-5)
- Were the examples clear? (1-5)

**Delivery:**
- Was the pacing good? (1-5)
- Was the instructor knowledgeable? (1-5)
- Were questions answered well? (1-5)

**Materials:**
- Were notebooks easy to follow? (1-5)
- Were hands-on exercises helpful? (1-5)
- Was the guide comprehensive? (1-5)

**Overall:**
- Would you recommend this workshop? (Yes/No)
- What was most valuable?
- What could be improved?
- What topics would you like to see added?

---

## 🙏 Acknowledgments

This workshop covers industry best practices drawn from:
- Databricks official documentation
- Apache Spark community guidelines
- Real-world production experience
- Community feedback and contributions

---

## 📄 License

These materials are provided for educational purposes.
Feel free to adapt and customize for your organization's needs.

---

**Version:** 1.0  
**Last Updated:** 2025-11-11  
**Maintained By:** [Your Team/Organization]  
**Contact:** [Contact Information]

---

## Quick Reference: Key Commands

```python
# Check execution plan
df.explain()
df.explain(mode='formatted')

# Check Spark UI
# Navigate to: Cluster → Spark UI → SQL/Jobs tabs

# Enable optimizations
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Broadcast hint
from pyspark.sql.functions import broadcast
df.join(broadcast(small_df), "key")

# Caching
df.cache()  # MEMORY_ONLY
df.persist(StorageLevel.MEMORY_AND_DISK)
df.unpersist()

# Partitioning
df.repartition(100)  # Increase partitions (with shuffle)
df.coalesce(10)      # Decrease partitions (no shuffle)

# Delta Lake operations
OPTIMIZE delta.`/path/to/table`
VACUUM delta.`/path/to/table` RETAIN 168 HOURS
DESCRIBE HISTORY delta.`/path/to/table`
```

---

**Ready to start? Open `Databricks_Performance_Pitfalls_Workshop.ipynb` and let's go! 🚀**
