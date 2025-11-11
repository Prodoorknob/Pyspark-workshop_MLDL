# 🚀 Databricks Free Edition Setup Guide

**Quick Start Guide for Workshop Participants**

---

## 📋 Table of Contents

1. [What is Databricks Free Edition?](#what-is-databricks-free-edition)
2. [Step-by-Step Setup](#step-by-step-setup)
3. [Creating Your First Cluster](#creating-your-first-cluster)
4. [Importing Notebooks](#importing-notebooks)
5. [Navigating the Databricks Workspace](#navigating-the-databricks-workspace)
6. [Using Built-in Datasets](#using-built-in-datasets)
7. [Tips & Tricks](#tips--tricks)
8. [Troubleshooting](#troubleshooting)
9. [FAQs](#faqs)

---

## 🎯 What is Databricks Free Edition?

**Databricks Free Edition** (also called Community Edition) is a completely free version of the Databricks platform that provides:

### What's Included ✅
- **Full PySpark functionality** (all DataFrame APIs, SQL, MLlib)
- **Single cluster** with 15 GB RAM and 2 cores
- **2 GB DBFS storage** (Databricks File System)
- **Pre-loaded datasets** (/databricks-datasets/)
- **Notebook environment** with collaboration features
- **MLflow tracking** (basic features)
- **Spark UI** for debugging and performance monitoring
- **Community support** via forums

### Limitations ⚠️
- **Single user only** (no team collaboration)
- **No job scheduling** (manual notebook execution only)
- **Clusters auto-terminate** after 2 hours of inactivity
- **No advanced features** (Delta sharing, Unity Catalog, SQL warehouses)
- **No SLA** (best effort support)

### Perfect For 🎯
- **Learning PySpark** and Spark fundamentals
- **Prototyping** data pipelines and ML models
- **Personal projects** and portfolio building
- **Workshops** and training sessions
- **Interview preparation** (hands-on practice)

---

## 🛠️ Step-by-Step Setup

### Step 1: Create Your Account (5 minutes)

1. **Go to the Databricks Community Edition signup page:**
   ```
   https://community.cloud.databricks.com/login.html
   ```
   Or directly to:
   ```
   https://www.databricks.com/try-databricks
   ```
   And select **"Get started with Community Edition"**

2. **Fill out the registration form:**
   - **Email address** (use personal or work email)
   - **First name**
   - **Last name**
   - **Password** (minimum 8 characters)
   - Check **"I agree to the terms of service"**

3. **Important:** Select **"Community Edition"** (NOT the free trial)
   - Look for the option that says "Get started with Community Edition"
   - This is 100% free with no credit card required

4. **Click "Sign Up"**

5. **Verify your email:**
   - Check your inbox for verification email from Databricks
   - Click the verification link
   - You'll be redirected to the Databricks workspace

### Step 2: First Login (2 minutes)

1. **Navigate to:**
   ```
   https://community.cloud.databricks.com/
   ```

2. **Log in with your credentials:**
   - Email
   - Password

3. **Welcome to Databricks! 🎉**
   You should see the Databricks workspace homepage.

---

## 🖥️ Creating Your First Cluster

### What is a Cluster?

A **cluster** is a set of computing resources (virtual machines) that run your Spark code. Think of it as your "engine" for processing data.

### Creating a Cluster (5 minutes)

1. **Navigate to Compute:**
   - Click **"Compute"** in the left sidebar
   - Or go to: `https://community.cloud.databricks.com/#setting/clusters`

2. **Click "Create Cluster"**

3. **Configure your cluster:**

   **Cluster Name:**
   ```
   workshop-cluster
   ```
   (You can name it anything, but "workshop-cluster" matches the tutorial)

   **Databricks Runtime Version:**
   - Select the **latest LTS ML version**
   - Example: **14.3 LTS ML** or **13.3 LTS ML**
   - **LTS = Long Term Support** (more stable)
   - **ML = Includes machine learning libraries** (MLflow, scikit-learn, etc.)

   **Node Type:**
   - This is pre-selected for Community Edition (cannot change)
   - You get: **15 GB RAM, 2 cores**

   **Terminate after:**
   - Default: **120 minutes (2 hours) of inactivity**
   - Cannot be changed in Community Edition
   - Cluster will stop if idle for 2 hours (notebook state saved!)

4. **Click "Create Cluster"**

5. **Wait for cluster to start (3-5 minutes):**
   - Status will show **"Pending"** → **"Starting"** → **"Running"**
   - You'll see a green dot when ready
   - Don't close the page during startup!

### Cluster States Explained

| State | What It Means | What To Do |
|-------|---------------|------------|
| **Pending** | Waiting for resources | Wait (usually <1 minute) |
| **Starting** | Launching virtual machines | Wait (3-5 minutes) |
| **Running** | Ready to use! | Start coding! |
| **Terminating** | Shutting down | Wait or restart if needed |
| **Terminated** | Stopped (saves money) | Click "Start" to restart |

---

## 📓 Importing Notebooks

### Option 1: Import from File (Recommended)

1. **Download notebooks from GitHub:**
   - `Part1_PySpark_Speedrun.ipynb`
   - `Part2_Data_Engineer_Toolkit.ipynb`
   - `Part3_ML_Capstone_Pipeline.ipynb`
   - `Part4_Real_Time_Prediction.ipynb`

2. **Navigate to Workspace:**
   - Click **"Workspace"** in the left sidebar
   - Click on **your username** (e.g., `Users/yourname@email.com`)

3. **Import each notebook:**
   - Click the **dropdown arrow** next to your username
   - Select **"Import"**
   - Choose **"File"**
   - Click **"Browse"** and select one of the `.ipynb` files
   - Click **"Import"**
   - Repeat for all 4 notebooks

4. **Verify import:**
   - You should see all 4 notebooks listed in your folder
   - Each should have a notebook icon 📓

### Option 2: Import from URL

If you have direct URLs to the notebooks:

1. **Workspace → Import**
2. Select **"URL"**
3. Paste the notebook URL
4. Click **"Import"**

### Option 3: Create and Copy-Paste

If import fails:

1. **Workspace → Create → Notebook**
2. Name it (e.g., "Part1_PySpark_Speedrun")
3. Select language: **Python**
4. Click **"Create"**
5. Copy-paste code from the original notebook cell by cell

---

## 🎓 Navigating the Databricks Workspace

### Main Navigation (Left Sidebar)

```
┌─────────────────────┐
│ 🏠 Home             │  Your landing page
│ 📁 Workspace        │  Notebooks and folders
│ 🔢 Repos            │  Git integration (not in CE)
│ 💻 Compute          │  Manage clusters
│ 🔬 Experiments      │  MLflow tracking
│ 📊 SQL              │  SQL editor (not in CE)
│ 📈 Dashboards       │  (Not in CE)
│ 💼 Jobs             │  (Not in CE)
│ ⚙️ Settings         │  User settings
└─────────────────────┘
```

### Notebook Interface

**Top Bar:**
```
┌──────────────────────────────────────────────────────┐
│ Notebook Name    [Cluster Dropdown]    Run All  ... │
└──────────────────────────────────────────────────────┘
```

**Key Buttons:**
- **Cluster Dropdown** (top right): Attach/detach cluster
- **Run All**: Execute all cells sequentially
- **Cell menu (▼)**: Run cell, add cell above/below, delete
- **Shift + Enter**: Run current cell and move to next

**Cell Types:**
- **Code cells** (default): Run Python, SQL, Scala, or R code
- **Markdown cells**: Documentation (click cell menu → "Change to Markdown")
- **Magic commands**: `%sql`, `%sh`, `%md`, `%fs`

### Important Views

1. **Spark UI:**
   - Click **"Compute"** → Your Cluster → **"Spark UI"**
   - View job execution, stages, tasks, storage
   - Essential for debugging and performance

2. **Experiments (MLflow):**
   - Click **"Experiments"** in left sidebar
   - View all ML runs, metrics, models
   - Appears after running ML code with MLflow

3. **DBFS (Databricks File System):**
   - Access via code: `dbutils.fs.ls("/path/")`
   - Or: Data tab (not always visible in CE)

---

## 📂 Using Built-in Datasets

Databricks Community Edition includes **pre-loaded datasets** at `/databricks-datasets/`. No need to download or upload data!

### Available Datasets

**List all datasets:**
```python
# Run this in a notebook cell
display(dbutils.fs.ls("/databricks-datasets/"))
```

**Popular datasets:**

| Dataset | Path | Description |
|---------|------|-------------|
| **TPC-H** | `/databricks-datasets/tpch/data-001/` | Business benchmark data (orders, customers) |
| **Airlines** | `/databricks-datasets/airlines/` | Flight delay data |
| **IoT** | `/databricks-datasets/iot/` | Sensor data |
| **Retail** | `/databricks-datasets/retail-org/` | E-commerce transactions |
| **MovieLens** | `/databricks-datasets/movielens/` | Movie ratings |
| **NYC Taxi** | `/databricks-datasets/nyctaxi/` | NYC taxi trip data |

### Loading Datasets (Examples)

**Load TPC-H orders (used in workshop):**
```python
df = spark.read.parquet("/databricks-datasets/tpch/data-001/orders.parquet")
df.show(5)
```

**Load CSV:**
```python
df = spark.read.csv("/databricks-datasets/airlines/part-00000", header=True, inferSchema=True)
```

**Load JSON:**
```python
df = spark.read.json("/databricks-datasets/iot/iot_devices.json")
```

**Explore dataset structure:**
```python
# List files in dataset directory
display(dbutils.fs.ls("/databricks-datasets/tpch/data-001/"))

# Check file size
dbutils.fs.ls("/databricks-datasets/tpch/data-001/orders.parquet")
```

---

## 💡 Tips & Tricks

### Productivity Tips

1. **Keyboard Shortcuts:**
   - `Shift + Enter`: Run cell and move to next
   - `Ctrl/Cmd + Enter`: Run cell and stay
   - `Esc + A`: Add cell above
   - `Esc + B`: Add cell below
   - `Esc + DD`: Delete cell
   - `Esc + M`: Convert cell to Markdown

2. **Magic Commands:**
   ```python
   %sql SELECT * FROM table_name;  # Run SQL
   %sh ls -la                       # Run shell command
   %fs ls /databricks-datasets/     # File system operations
   %md # Markdown text here         # Render markdown
   ```

3. **Display vs Print:**
   ```python
   print(df.show())   # ❌ Not ideal in Databricks
   display(df)        # ✅ Better: interactive table, charts
   ```

4. **Auto-complete:**
   - Press `Tab` after typing to see suggestions
   - Works for column names, functions, paths

5. **Code Formatting:**
   ```python
   # Format code automatically
   # Edit → Format Cell (or Ctrl+Shift+F)
   ```

### Performance Tips

1. **Cache frequently used DataFrames:**
   ```python
   df = spark.read.parquet("/path/to/data")
   df.cache()  # Keeps data in memory
   # Use df multiple times...
   df.unpersist()  # Release memory when done
   ```

2. **Limit data during development:**
   ```python
   # Work with sample during testing
   df_sample = df.limit(1000)
   ```

3. **Monitor with Spark UI:**
   - Check job duration
   - Identify slow stages
   - Spot data skew issues

4. **Avoid .collect() on large data:**
   ```python
   df.collect()  # ❌ Brings ALL data to driver (can crash)
   df.show(20)   # ✅ Shows sample only
   ```

### Collaboration Tips

1. **Export notebooks:**
   - Notebook menu → Export → DBC Archive (or .ipynb)
   - Share with others or save backup

2. **Version control:**
   - Save notebook versions: Notebook menu → Revision History
   - View and restore previous versions

3. **Comments in notebooks:**
   - Use markdown cells for documentation
   - Add code comments for complex logic

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Cluster Won't Start

**Symptoms:**
- Stuck on "Pending" or "Starting"
- Error: "Cloud provider error"

**Solutions:**
```
1. Wait 10 minutes (sometimes takes longer)
2. Refresh the page
3. Try creating a new cluster with a different name
4. Check Databricks status: https://status.databricks.com/
5. Try again later (resource constraints)
```

#### 2. Notebook Not Attached to Cluster

**Symptoms:**
- Cells won't run
- Top-right shows "Detached"

**Solutions:**
```
1. Click cluster dropdown (top-right)
2. Select your cluster name
3. Wait for "Attached" status
4. If cluster is terminated, click "Start"
```

#### 3. "Path Does Not Exist" Error

**Symptoms:**
```python
AnalysisException: Path does not exist: /databricks-datasets/...
```

**Solutions:**
```
1. Verify exact path (case-sensitive!):
   dbutils.fs.ls("/databricks-datasets/")
   
2. Check for typos in path

3. Ensure you're in Databricks (not local Jupyter)

4. Try absolute path: /databricks-datasets/tpch/data-001/orders.parquet
```

#### 4. "Cannot Resolve Column" Error

**Symptoms:**
```python
AnalysisException: cannot resolve 'column_name'
```

**Solutions:**
```
1. Check column name (case-sensitive):
   df.printSchema()
   
2. After joins, specify DataFrame:
   df1.join(df2, ...).select(df1.col1, df2.col2)
   
3. Use col() function:
   from pyspark.sql.functions import col
   df.select(col("column_name"))
```

#### 5. MLflow Experiments Not Showing

**Symptoms:**
- Can't find experiment in Experiments tab

**Solutions:**
```
1. Save notebook (Cmd+S / Ctrl+S)
2. Refresh page
3. Check notebook is in your user folder (not /Shared/)
4. Verify code ran successfully (check for errors)
5. Use programmatic search:
   import mlflow
   runs = mlflow.search_runs()
   print(runs)
```

#### 6. Out of Memory Error

**Symptoms:**
```
OutOfMemoryError: Java heap space
```

**Solutions:**
```
1. Use .limit() to reduce data size:
   df.limit(10000)
   
2. Repartition data:
   df.repartition(10)
   
3. Use .cache() judiciously (don't cache everything)

4. Process in smaller batches

5. Restart cluster (clears memory)
```

#### 7. Cluster Auto-Terminated

**Symptoms:**
- Cluster stopped after 2 hours of inactivity

**Solutions:**
```
1. This is normal in Community Edition
2. Notebook state is saved (variables lost)
3. Click "Start" to restart cluster
4. Re-run cells to restore state
5. Keep cluster active: run a cell every 1-2 hours
```

#### 8. Import Notebook Failed

**Symptoms:**
- Upload doesn't work
- Notebook appears corrupted

**Solutions:**
```
1. Verify file is .ipynb format (not .txt or .json)
2. Try different browser (Chrome recommended)
3. Use "Import" from URL if available
4. Create blank notebook and copy-paste cells manually
5. Check file isn't corrupted (open in text editor)
```

---

## ❓ FAQs

### General Questions

**Q: Is Databricks Community Edition really free?**
A: Yes! 100% free forever. No credit card required, no hidden fees.

**Q: What's the difference between Community Edition and Free Trial?**
A:
- **Community Edition**: Free forever, single user, basic features, 2-hour cluster timeout
- **Free Trial**: 14-day full access, team features, premium support, no cluster timeout

**Q: Can I use Community Edition for production workloads?**
A: No. Community Edition is for learning and prototyping only. Use paid Databricks for production.

**Q: How long does my account last?**
A: Forever! Community Edition accounts don't expire.

**Q: Can I upgrade from Community Edition to paid?**
A: Yes, but you'll need to create a new paid account. You can export/import notebooks.

### Data Questions

**Q: How much data can I process?**
A: Technically unlimited (constrained by cluster size). Realistically, up to a few GB runs smoothly.

**Q: Can I upload my own datasets?**
A: Yes! Use:
```python
# Upload via UI: Data → Add Data
# Or via code:
dbutils.fs.cp("file:/local/path", "dbfs:/uploaded/")
```

**Q: How much storage do I get?**
A: 2 GB in DBFS (Databricks File System). Use external storage (S3, Azure Blob) for more.

**Q: Where is my data stored?**
A: DBFS (backed by cloud storage, usually AWS S3). Exact location managed by Databricks.

### Cluster Questions

**Q: Can I create multiple clusters?**
A: Yes, but only **one can run at a time** in Community Edition.

**Q: Can I increase cluster size?**
A: No. Community Edition clusters are fixed at 15 GB RAM, 2 cores.

**Q: What happens when cluster terminates?**
A: Notebook state saved, but variables/data in memory lost. Restart cluster and re-run cells.

**Q: How do I keep cluster running longer?**
A: Run any cell every 1-2 hours (within 2-hour timeout). Or restart cluster as needed.

### Notebook Questions

**Q: How do I share notebooks with others?**
A: Export (Notebook menu → Export) and share file. Or share Databricks workspace link (requires recipient to have Databricks account).

**Q: Can I use Git with notebooks?**
A: Limited in Community Edition. Use export/import workflow, or upgrade to paid version for Git integration.

**Q: Do I lose my work if I close the browser?**
A: No! Notebooks auto-save. You can safely close and reopen.

**Q: Can I schedule notebooks to run automatically?**
A: No, not in Community Edition. Requires paid version with Jobs feature.

### MLflow Questions

**Q: Is MLflow included in Community Edition?**
A: Yes! Basic MLflow tracking works. Advanced features (model registry, serving) require paid version.

**Q: Where are my MLflow experiments stored?**
A: Within your Databricks workspace. Export models if you want to save them elsewhere.

**Q: Can I deploy models from Community Edition?**
A: Not directly. Export model and deploy elsewhere (e.g., Docker, cloud service).

### Troubleshooting Questions

**Q: Why is my cluster stuck on "Pending"?**
A: Usually resource constraints. Wait 10 mins, or try creating a new cluster. Check https://status.databricks.com/

**Q: Code runs in local Jupyter but not in Databricks. Why?**
A: Databricks uses different runtime. Check for:
- Library versions (may differ)
- File paths (use DBFS paths)
- Databricks-specific features (display(), dbutils)

**Q: How do I reset everything and start fresh?**
A: Delete notebooks, terminate cluster, restart cluster. Or create new cluster with different name.

---

## 📚 Additional Resources

### Official Documentation
- **Databricks Documentation**: https://docs.databricks.com/
- **Apache Spark Documentation**: https://spark.apache.org/docs/latest/
- **PySpark API Reference**: https://spark.apache.org/docs/latest/api/python/

### Learning Resources
- **Databricks Academy** (free courses): https://academy.databricks.com/
- **Databricks Community Forums**: https://community.databricks.com/
- **PySpark Cheat Sheet**: [PYSPARK_CHEATSHEET.md](PYSPARK_CHEATSHEET.md)

### Community
- **Databricks Community Edition Forum**: https://community.databricks.com/
- **Stack Overflow**: Tag `databricks` or `pyspark`
- **Reddit**: r/databricks, r/apachespark

### Video Tutorials
- **Databricks YouTube Channel**: https://www.youtube.com/c/Databricks
- Search: "Databricks Community Edition tutorial"
- Search: "PySpark tutorial for beginners"

---

## 🎯 Quick Reference Card

### Essential Commands
```python
# List datasets
dbutils.fs.ls("/databricks-datasets/")

# Load data
df = spark.read.parquet("/path/to/data.parquet")

# Show data
display(df)

# Check schema
df.printSchema()

# Save data
df.write.format("delta").save("/path/to/output")

# MLflow tracking
import mlflow
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.95)
```

### Essential Paths
```
Datasets: /databricks-datasets/
DBFS Root: /
Temp files: /tmp/
User folder: /Users/your.email@domain.com/
```

### Cluster States
```
Pending   → Waiting for resources
Starting  → Launching (3-5 mins)
Running   → Ready! ✅
Terminating → Shutting down
Terminated  → Stopped (restart if needed)
```

---

## 🚀 You're All Set!

Congratulations! You now have:
- ✅ Databricks Community Edition account
- ✅ Running cluster
- ✅ Imported notebooks
- ✅ Access to built-in datasets
- ✅ Knowledge to troubleshoot issues

**Ready to start the workshop! 🎉**

---

**Need help? Check the troubleshooting section or ask in the Databricks Community Forums!**

**Happy Learning! 🎓✨**
