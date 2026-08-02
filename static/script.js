const uploadForm = document.getElementById("uploadForm");
const videoInput = document.getElementById("videoFile");
const dropZone = document.getElementById("dropZone");

const loading = document.getElementById("loadingSection");
const result = document.getElementById("resultSection");

const predictionLabel = document.getElementById("predictionLabel");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");

// ----------------------
// Drag & Drop
// ----------------------

dropZone.addEventListener("click", () => {
    videoInput.click();
});

videoInput.addEventListener("change", () => {

    if(videoInput.files.length > 0){

        dropZone.innerHTML = `
            <h2>✅</h2>
            <h4>${videoInput.files[0].name}</h4>
            <p>${(videoInput.files[0].size/1024/1024).toFixed(2)} MB</p>
        `;
    }

});

dropZone.addEventListener("dragover",(e)=>{

    e.preventDefault();

    dropZone.classList.add("drag");

});

dropZone.addEventListener("dragleave",()=>{

    dropZone.classList.remove("drag");

});

dropZone.addEventListener("drop",(e)=>{

    e.preventDefault();

    dropZone.classList.remove("drag");

    if(e.dataTransfer.files.length>0){

        videoInput.files=e.dataTransfer.files;

        dropZone.innerHTML=`
            <h2>✅</h2>
            <h4>${e.dataTransfer.files[0].name}</h4>
            <p>${(e.dataTransfer.files[0].size/1024/1024).toFixed(2)} MB</p>
        `;

    }

});

// ----------------------
// Upload
// ----------------------

uploadForm.addEventListener("submit",async(e)=>{

    e.preventDefault();

    if(videoInput.files.length===0){

        alert("Please choose a video.");

        return;

    }

    const formData=new FormData();

    formData.append("file",videoInput.files[0]);

    loading.style.display="block";

    result.style.display="none";

    try{

        const response=await fetch("/api/predict",{

            method:"POST",

            body:formData

        });

        const data=await response.json();

        loading.style.display="none";

        if(data.error){

            alert(data.error);

            return;

        }

        result.style.display="block";

        predictionLabel.innerHTML=data.label;

        confidenceValue.innerHTML=data.confidence+" %";

        confidenceBar.style.width=data.confidence+"%";

        confidenceBar.innerHTML=data.confidence+"%";

        if(data.label==="FAKE"){

            predictionLabel.style.color="#ff3b3b";

            confidenceBar.classList.remove("bg-success");

            confidenceBar.classList.add("bg-danger");

        }

        else{

            predictionLabel.style.color="#00ff95";

            confidenceBar.classList.remove("bg-danger");

            confidenceBar.classList.add("bg-success");

        }

        result.scrollIntoView({

            behavior:"smooth"

        });

    }

    catch(error){

        loading.style.display="none";

        alert("Server Error");

        console.log(error);

    }

});